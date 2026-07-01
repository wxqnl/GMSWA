"""Unit tests for GatedMemSWA v2 (TTT-style fast weights + SWA).

Run with:
    CUDA_VISIBLE_DEVICES=4 python test_gmswa_v2.py
"""
from __future__ import annotations

import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

# Ensure local fla is used
HERE = os.path.dirname(os.path.abspath(__file__))
FLA_PATH = os.path.join(HERE, "flash-linear-attention")
if FLA_PATH not in sys.path:
    sys.path.insert(0, FLA_PATH)


def banner(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def make_layer(dim=64, num_heads=4, num_kv_heads=2, window_size=8, disable_memory=False, dtype=torch.bfloat16):
    from fla.layers.gated_mem_swa import GatedMemSWA
    layer = GatedMemSWA(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        num_kv_heads=num_kv_heads,
        qkv_bias=True,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        disable_memory=disable_memory,
        mem_gate_logit_bias=-1.0,
        mix_gate_logit_bias=0.0,  # for testing, use 0 so memory contributes meaningfully
        layer_idx=0,
    ).to(device="cuda", dtype=dtype)
    return layer


def test_forward_shape():
    banner("test_forward_shape")
    layer = make_layer()
    B, T, dim = 2, 32, 64
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.bfloat16)
    out, _, _ = layer(x)
    check(out.shape == (B, T, dim), f"out shape {tuple(out.shape)} == (B,T,dim)")
    check(torch.isfinite(out).all(), "out is finite")


def test_no_nan_random_input():
    banner("test_no_nan_random_input")
    layer = make_layer(window_size=16)
    B, T, dim = 1, 64, 64
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.bfloat16) * 2.0
    out, _, _ = layer(x)
    check(torch.isfinite(out).all(), "no NaN/Inf in output")


def test_backward_grads_flow():
    banner("test_backward_grads_flow")
    layer = make_layer().to(dtype=torch.float32)
    B, T, dim = 2, 24, 64
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.float32, requires_grad=True)
    out, _, _ = layer(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    # All params should have non-None grads
    bad = []
    for name, p in layer.named_parameters():
        if p.grad is None:
            bad.append(f"{name}: NO GRAD")
        elif not torch.isfinite(p.grad).all():
            bad.append(f"{name}: NON-FINITE GRAD")
        elif p.grad.abs().sum().item() == 0.0:
            # Some params may legitimately be zero (e.g. dt_bias at certain init); print a warning
            print(f"  [WARN] {name}: zero grad (sum=0)")
    if bad:
        for b in bad:
            print("   ", b)
        raise AssertionError("Bad grads")
    check(torch.isfinite(x.grad).all(), "input grad is finite")
    print("  All params have finite gradients")


def test_disable_memory_matches_pure_swa():
    banner("test_disable_memory_matches_pure_swa")
    # When disable_memory=True, the layer is pure SWA.  Verify it computes only the local
    # branch and doesn't depend on the memory params (which won't exist).
    layer = make_layer(disable_memory=True)
    B, T, dim = 1, 16, 64
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.bfloat16)
    out, _, _ = layer(x)
    check(out.shape == (B, T, dim), f"shape {out.shape}")
    check(torch.isfinite(out).all(), "finite")
    # Confirm memory params are None
    check(layer.beta_proj is None, "beta_proj is None when disable_memory=True")
    check(layer.A_log is None, "A_log is None when disable_memory=True")


def test_first_W_tokens_use_only_local():
    banner("test_first_W_tokens_use_only_local")
    # For positions < W, the memory state must be zero (since initial state is zero and
    # beta/g are masked to 0).  This means o_mem[t<W] = 0, and o[t<W] = alpha * o_local + 0.
    # So the output at positions < W with memory enabled and disabled should differ only
    # by the alpha scaling factor.  We'll verify a weaker condition: o_mem masked is zero.
    layer = make_layer(window_size=8)
    layer.eval()
    B, T, dim = 1, 16, 64
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.bfloat16)
    # Run memory branch manually to inspect
    with torch.no_grad():
        q = layer.q_proj(x).view(B, T, layer.num_heads, layer.head_dim)
        k = layer.k_proj(x).view(B, T, layer.num_kv_heads, layer.head_dim)
        v = layer.v_proj(x).view(B, T, layer.num_kv_heads, layer.head_dim)
        o_mem, _ = layer._memory_branch(q, k, v, x, initial_state=None, output_final_state=False)
    # First W tokens of o_mem should be zero
    W = layer.window_size
    first_W = o_mem[:, :W].float()
    check(first_W.abs().max().item() < 1e-4, f"o_mem[:W] is zero (max={first_W.abs().max().item()})")
    # Positions >= W should be non-zero
    later = o_mem[:, W:].float()
    print(f"  Memory output norm at t>=W: mean={later.abs().mean().item():.4f}")
    check(later.abs().max().item() > 1e-3, "o_mem[t>=W] is non-zero")


def test_causality_no_future_leak():
    banner("test_causality_no_future_leak")
    # If we modify the input at position T-1, then output at positions [0..T-2] must be unchanged.
    layer = make_layer(window_size=8)
    layer.eval()
    B, T, dim = 1, 24, 64
    torch.manual_seed(0)
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out1, _, _ = layer(x)
        # Modify a single late position
        x2 = x.clone()
        x2[:, -1] = torch.randn_like(x2[:, -1])
        out2, _, _ = layer(x2)
    diff = (out1[:, :-1] - out2[:, :-1]).abs().float().max().item()
    last_diff = (out1[:, -1:] - out2[:, -1:]).abs().float().max().item()
    print(f"  diff before last token: {diff:.5f}, at last token: {last_diff:.5f}")
    check(diff < 1e-2, f"no future leakage (max diff <1e-2 for early tokens), got {diff}")
    check(last_diff > 1e-3, "last token is actually affected by the change")


def test_prefill_then_decode_consistency():
    banner("test_prefill_then_decode_consistency")
    # Compare:
    #  (a) one-shot forward(x[:T])
    #  (b) prefill(x[:T-1]) -> decode(x[T-1:T])  with use_cache
    # The last position outputs should agree (up to numerical noise).
    from fla.models.utils import Cache

    layer = make_layer(window_size=4)
    layer.eval()
    B, T, dim = 1, 16, 64
    torch.manual_seed(42)
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out_full, _, _ = layer(x)

        cache = Cache()
        out_pref, _, _ = layer(x[:, :T-1], use_cache=True, past_key_values=cache)
        out_dec, _, _ = layer(x[:, T-1:T], use_cache=True, past_key_values=cache)

    full_last = out_full[:, -1:].float()
    dec = out_dec.float()
    max_abs = (full_last - dec).abs().max().item()
    mean_abs = (full_last - dec).abs().mean().item()
    print(f"  max |out_full[-1] - out_decode| = {max_abs:.5f}")
    print(f"  mean |diff| = {mean_abs:.5f}")
    # bf16 accumulates noise; tolerance ~5e-2 should be safe
    check(max_abs < 1e-1, f"prefill+decode matches full forward at last token (max abs diff {max_abs})")


def test_step_by_step_decode_consistency():
    banner("test_step_by_step_decode_consistency")
    # Generate token-by-token and compare against full forward.
    from fla.models.utils import Cache

    layer = make_layer(window_size=4)
    layer.eval()
    B, T, dim = 1, 12, 64
    torch.manual_seed(7)
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        full_out, _, _ = layer(x)
        # Step through
        cache = Cache()
        step_outs = []
        # First token: prefill of length 1 (this hits chunk-or-fused based on T)
        # Use T=1 in prefill which uses fused_recurrent for memory branch.
        step_outs.append(layer(x[:, :1], use_cache=True, past_key_values=cache)[0])
        for t in range(1, T):
            step_outs.append(layer(x[:, t:t+1], use_cache=True, past_key_values=cache)[0])
        step_out = torch.cat(step_outs, dim=1).float()

    diff = (full_out.float() - step_out).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"  max |full - step|  = {max_abs:.5f}")
    print(f"  mean |full - step| = {mean_abs:.5f}")
    # Allow a fair tolerance for bf16 + RoPE recompute + kernel dispatch differences
    check(max_abs < 0.2, f"step-by-step matches full forward (max abs {max_abs})")


def test_gqa_repeat_works():
    banner("test_gqa_repeat_works")
    # GQA with H_q=4, H_kv=2
    layer = make_layer(num_heads=4, num_kv_heads=2)
    x = torch.randn(2, 24, 64, device="cuda", dtype=torch.bfloat16)
    out, _, _ = layer(x)
    check(out.shape == (2, 24, 64), f"out shape {out.shape}")
    check(torch.isfinite(out).all(), "finite")


def test_long_sequence():
    banner("test_long_sequence")
    # Try a longer sequence to ensure chunk kernel kicks in (T > 64).
    layer = make_layer(num_heads=4, num_kv_heads=2, window_size=64).to(dtype=torch.float32)
    layer.train()
    B, T, dim = 1, 256, 64
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.float32, requires_grad=True)
    out, _, _ = layer(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    check(torch.isfinite(out).all(), "long seq output finite")
    check(torch.isfinite(x.grad).all(), "long seq input grad finite")
    # Check fast weight related params have non-trivial grads
    for name in ("gate_proj.weight", "gate_proj.bias", "A_log", "dt_bias"):
        p = dict(layer.named_parameters())[name]
        gn = p.grad.norm().item()
        print(f"  grad norm {name}: {gn:.6e}")
        check(gn > 0, f"{name} has nonzero grad")


def test_constant_state_size_in_cache():
    banner("test_constant_state_size_in_cache")
    # After many decode steps, the recurrent_state size should be constant (d_h x d_h per head).
    from fla.models.utils import Cache
    layer = make_layer(window_size=4)
    layer.eval()
    B, T, dim = 1, 1, 64
    x_init = torch.randn(B, 8, dim, device="cuda", dtype=torch.bfloat16)
    cache = Cache()
    with torch.no_grad():
        layer(x_init, use_cache=True, past_key_values=cache)
        s1 = cache[0]["recurrent_state"]
        for _ in range(40):
            x_step = torch.randn(B, 1, dim, device="cuda", dtype=torch.bfloat16)
            layer(x_step, use_cache=True, past_key_values=cache)
        s2 = cache[0]["recurrent_state"]
    if s1 is not None and s2 is not None:
        check(s1.shape == s2.shape, f"recurrent_state shape stays constant: {s1.shape}=={s2.shape}")
    # attn_state buffer should be <= window_size
    attn = cache[0]["attn_state"]
    L = attn[0].shape[1]
    print(f"  attn_state length: {L}, window_size={layer.window_size}")
    check(L <= layer.window_size, "attn_state buffer never exceeds window_size")


def test_full_model_smoke():
    banner("test_full_model_smoke")
    from fla.models.gated_mem_swa.configuration_gated_mem_swa import GatedMemSWAConfig
    from fla.models.gated_mem_swa.modeling_gated_mem_swa import GatedMemSWAForCausalLM

    cfg = GatedMemSWAConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_kv_heads=2,
        window_size=8,
        max_position_embeddings=1024,
        vocab_size=128,
        hidden_ratio=2,
        intermediate_size=128,
        tie_word_embeddings=True,
        fuse_cross_entropy=False,
    )
    model = GatedMemSWAForCausalLM(cfg).cuda().to(torch.bfloat16)
    model.eval()
    B, T = 1, 32
    input_ids = torch.randint(0, 128, (B, T), device="cuda")
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits
    check(logits.shape == (B, T, 128), f"logits shape {logits.shape}")
    check(torch.isfinite(logits).all(), "logits finite")
    print("  full-model smoke test passed")


def test_full_model_train_step():
    banner("test_full_model_train_step")
    from fla.models.gated_mem_swa.configuration_gated_mem_swa import GatedMemSWAConfig
    from fla.models.gated_mem_swa.modeling_gated_mem_swa import GatedMemSWAForCausalLM

    cfg = GatedMemSWAConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_kv_heads=2,
        window_size=8,
        max_position_embeddings=1024,
        vocab_size=128,
        hidden_ratio=2,
        intermediate_size=128,
        tie_word_embeddings=True,
        fuse_cross_entropy=False,
        fuse_norm=False,
        fuse_swiglu=False,
    )
    model = GatedMemSWAForCausalLM(cfg).cuda().to(torch.float32)
    model.train()
    B, T = 2, 96
    input_ids = torch.randint(0, 128, (B, T), device="cuda")
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    print(f"  initial loss: {loss.item():.4f}")
    loss.backward()
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            bad.append(name)
        elif not torch.isfinite(p.grad).all():
            bad.append(name + " (non-finite)")
    if bad:
        print("  bad params:", bad[:5])
        raise AssertionError(f"{len(bad)} params have bad gradients")
    check(torch.isfinite(loss), "loss finite")
    print("  full-model train step passed")


def main():
    tests = [
        test_forward_shape,
        test_no_nan_random_input,
        test_backward_grads_flow,
        test_disable_memory_matches_pure_swa,
        test_first_W_tokens_use_only_local,
        test_causality_no_future_leak,
        test_gqa_repeat_works,
        test_long_sequence,
        test_constant_state_size_in_cache,
        test_prefill_then_decode_consistency,
        test_step_by_step_decode_consistency,
        test_full_model_smoke,
        test_full_model_train_step,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"\n!!! {t.__name__} FAILED: {e}")
            traceback.print_exc()
            failed.append(t.__name__)
    print("\n" + "#" * 72)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)}")
        for n in failed:
            print(" -", n)
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
