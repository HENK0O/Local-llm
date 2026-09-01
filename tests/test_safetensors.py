import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_llm.safetensors import SafeTensorError, load_directory, load_file, save_file


def write_safetensors(path: Path, tensors):
    header = {}
    payload = bytearray()
    dtype_names = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("uint16"): "BF16"}
    for name, array in tensors.items():
        array = np.asarray(array)
        begin = len(payload)
        payload.extend(array.tobytes())
        header[name] = {"dtype": dtype_names[array.dtype], "shape": list(array.shape),
                        "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


class SafeTensorTests(unittest.TestCase):
    def test_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "written.safetensors"
            expected = {
                "half": np.arange(6, dtype=np.float16).reshape(2, 3),
                "float": np.array([1.25, -2.5], dtype=np.float32),
            }
            save_file(path, expected, {"source": "test"})
            actual = load_file(path, np.float16)
            for name, value in expected.items():
                np.testing.assert_array_equal(actual[name], value)

    def test_f32_is_memory_mapped_and_f16_is_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            write_safetensors(path, {
                "float": np.arange(6, dtype=np.float32).reshape(2, 3),
                "half": np.array([1.5, -2.0], dtype=np.float16),
            })
            tensors = load_file(path)
            self.assertIsInstance(tensors["float"], np.memmap)
            self.assertEqual(tensors["half"].dtype, np.float32)
            np.testing.assert_array_equal(tensors["float"], np.arange(6).reshape(2, 3))
            np.testing.assert_allclose(tensors["half"], [1.5, -2.0])

    def test_bfloat16_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            values = np.array([1.0, -2.5, 0.25], dtype=np.float32)
            words = (values.view(np.uint32) >> 16).astype(np.uint16)
            write_safetensors(path, {"bf16": words})
            np.testing.assert_array_equal(load_file(path)["bf16"], values)

    def test_sharded_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_safetensors(root / "part-1.safetensors", {"a": np.array([1], dtype=np.float32)})
            write_safetensors(root / "part-2.safetensors", {"b": np.array([2], dtype=np.float32)})
            with (root / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
                json.dump({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}, handle)
            tensors = load_directory(root)
            self.assertEqual(set(tensors), {"a", "b"})

    def test_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            with path.open("wb") as handle:
                handle.write(struct.pack("<Q", 100))
            with self.assertRaises(SafeTensorError):
                load_file(path)


if __name__ == "__main__":
    unittest.main()
