"""100-step synthetic-data training smoke test for GM-SWA v2.

Run:
    CUDA_VISIBLE_DEVICES=0 .venv311/bin/python smoke_train_v2.py [--scale 340M|1B|small]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flash-linear-attention"))

import torch
from transformers import AutoConfig

import fla  # registers gated_mem_swa
from fla.models.gated_mem_swa.modeling_gated_mem_swa import GatedMemSWAForCausalLM


def make_model(scale: str) -> tuple[torch.nn.Module, int]:
    if scale == "small":
        cfg = AutoConfig.from_pretrained("flash-linear-attention/flame/configs/gated_mem_swa_340M.json")
        cfg.num_hidden_layers = 4
        cfg.hidden_size = 256
        cfg.num_heads = 8
        cfg.num_kv_heads = 2
        cfg.intermediate_size = 1024
        cfg.window_size = 64
        cfg.max_position_embeddings = 2048
    elif scale == "340M":
        cfg = AutoConfig.from_pretrained("flash-linear-attention/flame/configs/gated_mem_swa_340M.json")
    elif scale == "1B":
        cfg = AutoConfig.from_pretrained("flash-linear-attention/flame/configs/gated_mem_swa_1B.json")
    else:
        raise ValueError(scale)
    cfg.fuse_norm = False
    cfg.fuse_swiglu = False
    cfg.fuse_cross_entropy = False
    model = GatedMemSWAForCausalLM(cfg).cuda().to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small", choices=["small", "340M", "1B"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    model, n_params = make_model(args.scale)
    print(f"[smoke] scale={args.scale}, params={n_params/1e6:.1f}M, seq_len={args.seq_len}, batch={args.batch}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-15)

    V = model.config.vocab_size
    losses = []
    # Fixed batch for overfit smoke. Real signal: the model SHOULD memorize this.
    torch.manual_seed(1)
    fixed_ids = torch.randint(0, V, (args.batch, args.seq_len), device="cuda")
    t0 = time.time()
    for step in range(args.steps):
        ids = fixed_ids
        out = model(input_ids=ids, labels=ids.clone())
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # check finite grads
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"  [step {step}] non-finite grad in {name}!")
                sys.exit(2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if step % 10 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            print(f"  step {step:4d} | loss = {loss.item():.4f} | elapsed = {elapsed:.1f}s")

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    print(f"[smoke] mean(initial 5) = {initial:.4f}, mean(final 5) = {final:.4f}, delta = {initial - final:+.4f}")
    assert final < initial - 0.1, f"loss did not decrease (initial={initial}, final={final})"
    print("[smoke] PASS: loss decreased and all grads finite")


if __name__ == "__main__":
    main()
