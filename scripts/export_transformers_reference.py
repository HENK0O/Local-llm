#!/usr/bin/env python3
"""Export reference activations. This script is not used by the runtime."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32
    ).eval()
    encoded = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids
    captured = {"input_ids": input_ids[0].numpy()}
    hooks = []

    def save(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().cpu().float().numpy()[0]
        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(save("embeddings")))
    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(save(f"layer.{index}")))
    hooks.append(model.model.norm.register_forward_hook(save("norm")))
    with torch.no_grad():
        captured["logits"] = model(input_ids, attention_mask=encoded.attention_mask).logits[0].cpu().float().numpy()
    for hook in hooks:
        hook.remove()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=encoded.attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0, input_ids.shape[1]:]
        captured["greedy_tokens"] = generated.cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **captured)
    print(f"Reference written to {args.output} ({len(input_ids[0])} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
