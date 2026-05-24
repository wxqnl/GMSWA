#!/usr/bin/env python
"""
Wrapper script to evaluate qwen2_ultralight model with lm_eval.
This ensures the model is properly registered before evaluation.
Supports multi-GPU evaluation via torchrun.
"""

import sys
import os

# Suppress verbose logging
import logging
logging.getLogger("lm_eval").setLevel(logging.WARNING)

# Add flash-linear-attention to path
sys.path.insert(0, '/home/user01/Minko/GMSWA/flash-linear-attention')

# Import fla to register the custom model
import fla  # noqa: F401


# Now run lm_eval from the local directory
if __name__ == "__main__":
    import torch.distributed as dist
    import torch
    import json

    # Initialize distributed if available
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    from lm_eval import evaluator, simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model_args = {
        "pretrained": "/home/user01/Minko/GMSWA/flash-linear-attention/flame/saves/GMswa-3-4slots-40B-stage1",
        "dtype": "bfloat16",
        "device": f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    }

    lm = HFLM(**model_args)

    # Only print on rank 0
    if local_rank == 0:
        print("=" * 60)
        print("Starting evaluation...")
        print("=" * 60)

    results = simple_evaluate(
        model=lm,
        tasks=["piqa","openbookqa","hellaswag","arc_easy","arc_challenge","wikitext"],
        #tasks=["piqa"],
        #batch_size=2400,
        batch_size="auto:4",
        max_batch_size=4096,
        #verbosity="DEBUG"  # 添加这行
    )

    if local_rank == 0:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Print only the scores
        if "results" in results:
            for task, scores in results["results"].items():
                print(f"\n[{task}]")
                for metric, value in scores.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")

        # Save full results to file
        output_path = "/home/user01/Minko/GMSWA/flash-linear-attention/flame/saves/GMswa-3-4slots-40B-stage1/results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        print("=" * 60)

    if world_size > 1:
        dist.destroy_process_group()
