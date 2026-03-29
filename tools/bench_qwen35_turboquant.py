#!/usr/bin/env python3

import argparse
import time

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Qwen3.5 TurboQuant backend in vLLM."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--mode", choices=("turbo4", "turbo3"), default="turbo4")
    parser.add_argument("--prompt", default="Summarize the image in one short sentence.")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    init_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        load_format=args.load_format,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_config={
            "turboquant_enabled": True,
            "turboquant_mode": args.mode,
        },
        enforce_eager=True,
    )
    init_elapsed = time.perf_counter() - init_start

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    gen_start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params)
    gen_elapsed = time.perf_counter() - gen_start

    text = outputs[0].outputs[0].text
    token_count = len(outputs[0].outputs[0].token_ids)
    tok_per_s = token_count / gen_elapsed if gen_elapsed > 0 else 0.0

    print(f"mode={args.mode}")
    print(f"model={args.model}")
    print(f"load_format={args.load_format}")
    print(f"init_seconds={init_elapsed:.3f}")
    print(f"generate_seconds={gen_elapsed:.3f}")
    print(f"generated_tokens={token_count}")
    print(f"decode_tok_per_s={tok_per_s:.3f}")
    print(f"output={text!r}")


if __name__ == "__main__":
    main()
