#!/usr/bin/env python3

import os

from vllm import LLM, SamplingParams


def main() -> None:
    mode = os.environ.get("TURBOQUANT_MODE", "turbo4")
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8"))
    llm = LLM(
        model="Qwen/Qwen3.5-4B",
        trust_remote_code=True,
        load_format="dummy",
        max_model_len=512,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_config={
            "turboquant_enabled": True,
            "turboquant_mode": mode,
        },
        enforce_eager=True,
    )
    print("llm_init_ok", type(llm).__name__, mode)

    outputs = llm.generate(
        ["hello"],
        SamplingParams(max_tokens=4, temperature=0.0),
    )
    print("generate_ok", len(outputs), len(outputs[0].outputs))
    print("text", outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
