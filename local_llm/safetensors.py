"""Minimal, dependency-free SafeTensors reader backed by NumPy memory maps."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


_DTYPES: Mapping[str, Tuple[np.dtype, int]] = {
    "BOOL": (np.dtype("?"), 1), "U8": (np.dtype("u1"), 1), "I8": (np.dtype("i1"), 1),
    "I16": (np.dtype("<i2"), 2), "U16": (np.dtype("<u2"), 2),
    "I32": (np.dtype("<i4"), 4), "U32": (np.dtype("<u4"), 4),
    "I64": (np.dtype("<i8"), 8), "U64": (np.dtype("<u8"), 8),
    "F16": (np.dtype("<f2"), 2), "BF16": (np.dtype("<u2"), 2),
    "F32": (np.dtype("<f4"), 4), "F64": (np.dtype("<f8"), 8),
}


class SafeTensorError(ValueError):
    pass


def _read_header(path: Path) -> Tuple[dict, int, int]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise SafeTensorError(f"{path}: file is shorter than a SafeTensors header")
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        if header_length > file_size - 8:
            raise SafeTensorError(f"{path}: header extends beyond end of file")
        try:
            header = json.loads(handle.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafeTensorError(f"{path}: invalid JSON header") from exc
    if not isinstance(header, dict):
        raise SafeTensorError(f"{path}: header must be a JSON object")
    return header, 8 + header_length, file_size


def _bfloat16_to_float32(raw: NDArray[np.uint16]) -> NDArray[np.float32]:
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)


def load_file(path: Path, float_dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    """Load one file; F32 tensors remain zero-copy memory-map views."""
    path = Path(path)
    header, data_start, file_size = _read_header(path)
    tensors: Dict[str, np.ndarray] = {}
    intervals = []
    data_size = file_size - data_start
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(info, dict):
            raise SafeTensorError(f"{path}: malformed tensor entry")
        dtype_name, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype_name not in _DTYPES:
            raise SafeTensorError(f"{path}: unsupported dtype {dtype_name!r} for {name}")
        if not isinstance(shape, list) or not all(isinstance(value, int) and value >= 0 for value in shape):
            raise SafeTensorError(f"{path}: invalid shape for {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(v, int) for v in offsets):
            raise SafeTensorError(f"{path}: invalid offsets for {name}")
        begin, end = offsets
        numpy_dtype, item_size = _DTYPES[dtype_name]
        if begin < 0 or end < begin or end > data_size or end - begin != math.prod(shape) * item_size:
            raise SafeTensorError(f"{path}: byte range does not match shape for {name}")
        intervals.append((begin, end, name))
        if end == begin:
            raw = np.empty(tuple(shape), dtype=numpy_dtype)
        else:
            raw = np.memmap(path, mode="r", dtype=numpy_dtype, offset=data_start + begin, shape=tuple(shape))
        if dtype_name == "BF16":
            tensor = _bfloat16_to_float32(raw)
        elif dtype_name in {"F16", "F32", "F64"} and np.dtype(float_dtype) != raw.dtype:
            tensor = raw.astype(float_dtype)
        else:
            tensor = raw
        tensors[name] = tensor
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise SafeTensorError(f"{path}: tensors {previous[2]} and {current[2]} overlap")
    return tensors


def _files_from_index(model_dir: Path, index_path: Path) -> Iterable[Path]:
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SafeTensorError(f"{index_path}: missing weight_map")
    names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and Path(name).name == name for name in names):
        raise SafeTensorError(f"{index_path}: unsafe shard name")
    return (model_dir / name for name in names)


def load_directory(model_dir: Path, float_dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    single, index = model_dir / "model.safetensors", model_dir / "model.safetensors.index.json"
    if single.exists():
        files: Iterable[Path] = [single]
    elif index.exists():
        files = _files_from_index(model_dir, index)
    else:
        raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {model_dir}")
    tensors: Dict[str, np.ndarray] = {}
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"missing SafeTensors shard {path}")
        for name, tensor in load_file(path, float_dtype).items():
            if name in tensors:
                raise SafeTensorError(f"duplicate tensor {name} across shards")
            tensors[name] = tensor
    return tensors


def save_file(
    path: Path,
    tensors: Mapping[str, np.ndarray],
    metadata: Optional[Mapping[str, str]] = None,
) -> None:
    """Write a deterministic SafeTensors file without the external package."""
    path = Path(path)
    if not tensors:
        raise SafeTensorError("cannot write an empty SafeTensors file")
    dtype_names = {
        np.dtype("bool"): "BOOL", np.dtype("uint8"): "U8", np.dtype("int8"): "I8",
        np.dtype("int16"): "I16", np.dtype("uint16"): "U16",
        np.dtype("int32"): "I32", np.dtype("uint32"): "U32",
        np.dtype("int64"): "I64", np.dtype("uint64"): "U64",
        np.dtype("float16"): "F16", np.dtype("float32"): "F32",
        np.dtype("float64"): "F64",
    }
    arrays: Dict[str, np.ndarray] = {}
    header: Dict[str, object] = {}
    offset = 0
    for name in sorted(tensors):
        if not isinstance(name, str) or name == "__metadata__":
            raise SafeTensorError(f"invalid tensor name {name!r}")
        array = np.ascontiguousarray(tensors[name])
        dtype = array.dtype.newbyteorder("=")
        if dtype not in dtype_names:
            raise SafeTensorError(f"unsupported dtype {array.dtype} for {name}")
        if array.dtype.byteorder == ">":
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        size = array.nbytes
        arrays[name] = array
        header[name] = {
            "dtype": dtype_names[dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if metadata is not None:
        if not all(isinstance(key, str) and isinstance(value, str)
                   for key, value in metadata.items()):
            raise SafeTensorError("SafeTensors metadata keys and values must be strings")
        header["__metadata__"] = dict(metadata)
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name in sorted(arrays):
            handle.write(arrays[name].tobytes(order="C"))
    temporary.replace(path)
