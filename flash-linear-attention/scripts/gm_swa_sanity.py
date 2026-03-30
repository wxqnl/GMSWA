import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fla.layers.gated_mem_swa import GatedMemSWA, repeat_kv


class SimpleSWA(nn.Module):
    def __init__(self, base: GatedMemSWA) -> None:
        super().__init__()
        self.q_proj = base.q_proj
        self.k_proj = base.k_proj
        self.v_proj = base.v_proj
        self.o_proj = base.o_proj
        self.rotary = base.rotary
        self.window_size = base.window_size
        self.num_heads = base.num_heads
        self.num_kv_heads = base.num_kv_heads
        self.num_kv_groups = base.num_kv_groups
        self.head_dim = base.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self.rotary(q, k, seqlen_offset=0, max_seqlen=seq_len)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        window_k = []
        window_v = []
        outputs = []
        for t in range(seq_len):
            if len(window_k) >= self.window_size:
                window_k.pop(0)
                window_v.pop(0)
            window_k.append(k[:, t])
            window_v.append(v[:, t])

            k_window = torch.stack(window_k, dim=2)
            v_window = torch.stack(window_v, dim=2)
            q_t = q[:, t].unsqueeze(2)
            attn_out = F.scaled_dot_product_attention(q_t, k_window, v_window, is_causal=False)
            attn_out = attn_out.squeeze(2).reshape(batch_size, 1, -1)
            outputs.append(self.o_proj(attn_out))

        return torch.cat(outputs, dim=1)


def main() -> None:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("gm_swa_sanity.py requires CUDA in this environment.")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    dim = 64
    num_heads = 4
    window_size = 4
    seq_len = 2 * window_size

    gm_swa = GatedMemSWA(dim=dim, num_heads=num_heads, window_size=window_size).to(device=device, dtype=dtype)
    gm_swa.eval()
    with torch.no_grad():
        gm_swa.gate_net.weight.zero_()
        gm_swa.gate_net.bias.zero_()

    swa = SimpleSWA(gm_swa)
    swa.eval()

    x = torch.randn(1, seq_len, dim, device=device, dtype=dtype)
    x2 = x.clone()
    x2[:, 0] += 1.0

    with torch.no_grad():
        out_gm = gm_swa(x)[0]
        out_gm2 = gm_swa(x2)[0]
        out_swa = swa(x)
        out_swa2 = swa(x2)
        kv_cache = None
        memory_state = None
        cached = []
        for t in range(seq_len):
            step_out, kv_cache, memory_state = gm_swa.inference_step(x[:, t : t + 1], kv_cache, memory_state)
            cached.append(step_out)
        out_cached = torch.cat(cached, dim=1)

    diff_gm = (out_gm[:, window_size:] - out_gm2[:, window_size:]).abs().mean().item()
    diff_swa = (out_swa[:, window_size:] - out_swa2[:, window_size:]).abs().mean().item()
    diff_cached = (out_gm - out_cached).abs().max().item()

    print(f"Device: {device} ({dtype})")
    print(f"GM-SWA diff (t >= window): {diff_gm:.6f}")
    print(f"SWA diff   (t >= window): {diff_swa:.6f}")
    print(f"GM-SWA cached/full max diff: {diff_cached:.6f}")


if __name__ == "__main__":
    main()
