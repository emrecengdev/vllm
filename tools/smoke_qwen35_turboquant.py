#!/usr/bin/env python3

import os

from vllm import LLM, SamplingParams


def env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def main() -> None:
    mode = os.environ.get("TURBOQUANT_MODE", "turbo4")
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8"))
    attention_config = {
        "turboquant_enabled": True,
        "turboquant_mode": mode,
        "turboquant_sparse_v": env_flag("TURBOQUANT_SPARSE_V"),
        "turboquant_layer_adaptive": env_flag("TURBOQUANT_LAYER_ADAPTIVE"),
    }
    llm = LLM(
        model="Qwen/Qwen3.5-4B",
        trust_remote_code=True,
        load_format="dummy",
        max_model_len=512,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_config=attention_config,
        enforce_eager=True,
    )
    print("llm_init_ok", type(llm).__name__, mode, attention_config)

    outputs = llm.generate(
        ["hello"],
        SamplingParams(max_tokens=4, temperature=0.0),
    )
    print("generate_ok", len(outputs), len(outputs[0].outputs))
    print("text", outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
