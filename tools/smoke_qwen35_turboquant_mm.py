#!/usr/bin/env python3

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from PIL import Image

from vllm import LLM, SamplingParams


def env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def build_image() -> Image.Image:
    image = Image.new("RGB", (96, 96), color=(240, 240, 240))
    for x in range(24, 72):
        for y in range(24, 72):
            image.putpixel((x, y), (220, 40, 40))
    return image


def main() -> None:
    model = os.environ.get("TURBOQUANT_MODEL", "Qwen/Qwen3.5-4B")
    mode = os.environ.get("TURBOQUANT_MODE", "turbo4")
    load_format = os.environ.get("TURBOQUANT_LOAD_FORMAT", "dummy")
    max_model_len = int(os.environ.get("TURBOQUANT_MAX_MODEL_LEN", "512"))
    tensor_parallel_size = int(os.environ.get("TURBOQUANT_TENSOR_PARALLEL_SIZE", "1"))
    disable_custom_all_reduce = env_flag("TURBOQUANT_DISABLE_CUSTOM_ALL_REDUCE")
    gpu_memory_utilization = float(
        os.environ.get("TURBOQUANT_GPU_MEMORY_UTILIZATION", "0.5")
    )
    attention_config = {
        "turboquant_enabled": True,
        "turboquant_mode": mode,
        "turboquant_sparse_v": env_flag("TURBOQUANT_SPARSE_V"),
        "turboquant_layer_adaptive": env_flag("TURBOQUANT_LAYER_ADAPTIVE"),
    }
    llm = LLM(
        model=model,
        trust_remote_code=True,
        load_format=load_format,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        disable_custom_all_reduce=disable_custom_all_reduce,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_config=attention_config,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    print(
        "llm_init_ok",
        type(llm).__name__,
        model,
        mode,
        load_format,
        tensor_parallel_size,
        disable_custom_all_reduce,
        attention_config,
    )

    tokenizer = llm.get_tokenizer()
    chat_prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image briefly."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = {
        "prompt": chat_prompt,
        "multi_modal_data": {"image": build_image()},
    }
    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    print("generate_ok", len(outputs), len(outputs[0].outputs))
    print("text", outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
