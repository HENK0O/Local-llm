from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from .evaluation import _load_baguette_module
from .generation import GenerationStats, sample_token


class ReferenceRuntime:
    """Lazy PyTorch/Transformers backend used only as an explicit reference."""

    def __init__(self, path: Path, repo: Optional[Path], expected_config) -> None:
        self.path = Path(path)
        self.repo = Path(repo) if repo is not None else None
        self.expected_config = expected_config
        self._model = None
        self._kind = "transformers" if self.path.is_dir() else "baguette"

    @property
    def name(self) -> str:
        prefix = "Transformers" if self._kind == "transformers" else "PyTorch"
        return f"{prefix} · {self.path.name}"

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "reference mode requires PyTorch; install local-llm[reference]"
            ) from exc

        if self._kind == "transformers":
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "a Hugging Face reference requires Transformers; "
                    "install local-llm[reference]"
                ) from exc
            model = AutoModelForCausalLM.from_pretrained(
                self.path, local_files_only=True, torch_dtype=torch.float32
            )
            vocab_size = int(model.config.vocab_size)
        else:
            repo = self.repo if self.repo is not None else self.path.parent
            module = _load_baguette_module(repo)
            data = torch.load(self.path, map_location="cpu", weights_only=True)
            if not isinstance(data, dict) or not isinstance(data.get("model_cfg"), dict):
                raise ValueError("invalid Baguette checkpoint")
            config = module.ModelConfig.from_dict(data["model_cfg"])
            model = module.build_model(config)
            model.load_state_dict(data["model"])
            vocab_size = int(config.vocab_size)

        if vocab_size != self.expected_config.vocab_size:
            raise ValueError("reference vocabulary does not match the local runtime model")
        model.eval()
        self._model = model
        return model

    @staticmethod
    def _tensor_bytes(value) -> int:
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            return int(value.numel() * value.element_size())
        if hasattr(value, "to_legacy_cache"):
            value = value.to_legacy_cache()
        if isinstance(value, dict):
            return sum(ReferenceRuntime._tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(ReferenceRuntime._tensor_bytes(item) for item in value)
        return 0

    def generate_tokens(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        seed: Optional[int],
    ) -> Iterator[Tuple[int, Optional[GenerationStats]]]:
        if self._kind == "transformers":
            yield from self._generate_transformers(
                prompt_tokens, max_new_tokens, temperature, top_k, top_p, seed
            )
        else:
            yield from self._generate_baguette(
                prompt_tokens, max_new_tokens, temperature, top_k, top_p, seed
            )

    def _generate_baguette(
        self, prompt_tokens, max_new_tokens, temperature, top_k, top_p, seed
    ):
        import torch

        model = self._load()
        maximum = int(model.cfg.max_seq_len) - len(prompt_tokens) + 1
        if max_new_tokens > maximum:
            raise ValueError(
                f"requested {max_new_tokens} new tokens, but only {max(0, maximum)} "
                "fit in the reference context"
            )
        tensor = torch.tensor([prompt_tokens], dtype=torch.long)
        capacity = min(int(model.cfg.max_seq_len), len(prompt_tokens) + max_new_tokens)
        dtype = next(model.parameters()).dtype
        caches = model._alloc_caches(1, max(capacity, 1), tensor.device, dtype)
        rng = np.random.default_rng(seed)
        emitted = []
        decode_seconds = 0.0

        with torch.inference_mode():
            start = time.perf_counter()
            current = model._forward_cached(tensor, caches, 0)[0, -1]
            prefill_seconds = time.perf_counter() - start
            position = len(prompt_tokens)
            for _ in range(max_new_tokens):
                token = sample_token(
                    current.float().cpu().numpy(), temperature, top_k, top_p, rng
                )
                emitted.append(token)
                finished = (
                    token == self.expected_config.eos_token_id
                    or len(emitted) == max_new_tokens
                )
                if finished:
                    yield token, GenerationStats(
                        len(prompt_tokens), len(emitted), prefill_seconds, decode_seconds,
                        self._tensor_bytes(caches),
                    )
                    return
                start = time.perf_counter()
                next_id = torch.tensor([[token]], dtype=torch.long)
                current = model._forward_cached(next_id, caches, position)[0, -1]
                decode_seconds += time.perf_counter() - start
                position += 1
                yield token, None

    def _generate_transformers(
        self, prompt_tokens, max_new_tokens, temperature, top_k, top_p, seed
    ):
        import torch

        model = self._load()
        context = int(getattr(model.config, "max_position_embeddings",
                              self.expected_config.max_position_embeddings))
        maximum = context - len(prompt_tokens) + 1
        if max_new_tokens > maximum:
            raise ValueError(
                f"requested {max_new_tokens} new tokens, but only {max(0, maximum)} "
                "fit in the reference context"
            )
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        rng = np.random.default_rng(seed)
        emitted = []
        decode_seconds = 0.0

        with torch.inference_mode():
            start = time.perf_counter()
            output = model(input_ids=input_ids, use_cache=True, return_dict=True)
            prefill_seconds = time.perf_counter() - start
            current = output.logits[0, -1]
            cache = output.past_key_values
            for _ in range(max_new_tokens):
                token = sample_token(
                    current.float().cpu().numpy(), temperature, top_k, top_p, rng
                )
                emitted.append(token)
                finished = (
                    token == self.expected_config.eos_token_id
                    or len(emitted) == max_new_tokens
                )
                if finished:
                    yield token, GenerationStats(
                        len(prompt_tokens), len(emitted), prefill_seconds, decode_seconds,
                        self._tensor_bytes(cache),
                    )
                    return
                start = time.perf_counter()
                output = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                decode_seconds += time.perf_counter() - start
                current = output.logits[0, -1]
                cache = output.past_key_values
                yield token, None
