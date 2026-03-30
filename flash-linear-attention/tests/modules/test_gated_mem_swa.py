import torch
import torch.nn as nn

from fla.layers.gated_mem_swa import flash_attn_varlen_func
from fla.models.gated_mem_swa.configuration_gated_mem_swa import GatedMemSWAConfig
from fla.models.gated_mem_swa.modeling_gated_mem_swa import GatedMemSWAForCausalLM


def build_config(**overrides) -> GatedMemSWAConfig:
    config = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_kv_heads=2,
        window_size=8,
        intermediate_size=128,
        vocab_size=256,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        gate_bias_init=1.0,
        mem_norm=True,
    )
    config.update(overrides)
    return GatedMemSWAConfig(**config)


def test_gate_initialization_survives_post_init():
    model = GatedMemSWAForCausalLM(build_config())
    attn = model.model.layers[0].attn

    assert torch.count_nonzero(attn.gate_net.weight) == 0
    assert torch.allclose(attn.gate_net.bias, torch.ones_like(attn.gate_net.bias))

    gates = attn._compute_gate(torch.zeros(2, 3, attn.dim))
    expected = torch.full_like(gates, torch.sigmoid(torch.tensor(1.0, dtype=gates.dtype)))
    assert torch.allclose(gates, expected, atol=1e-6, rtol=0)


def test_projected_memory_updates_are_bounded():
    model = GatedMemSWAForCausalLM(build_config(num_mem_slots=4, mem_update_source="value"))
    attn = model.model.layers[0].attn
    evicted_v = torch.randn(2, 16, attn.num_kv_heads, attn.head_dim) * 10

    updates = attn._project_memory_update(None, evicted_v)
    norms = updates.float().norm(dim=-1)

    assert torch.all(norms <= 1.0001)


def test_recurrent_memory_state_stays_bounded():
    model = GatedMemSWAForCausalLM(build_config(num_mem_slots=4, mem_update_source="value"))
    attn = model.model.layers[0].attn
    memory_state = torch.zeros(2, attn.num_kv_heads, attn.num_mem_slots, attn.head_dim)

    for _ in range(64):
        evicted_v = torch.randn(2, attn.num_kv_heads, attn.head_dim) * 10
        gate_input = torch.randn(2, attn.dim)
        memory_state = attn._update_memory(memory_state, None, evicted_v, gate_input)
        norms = memory_state.float().norm(dim=-1)
        assert torch.all(norms <= 1.0001)


def test_kv_memory_update_preserves_source_gradients():
    model = GatedMemSWAForCausalLM(build_config(mem_update_source="kv"))
    attn = model.model.layers[0].attn.float()
    evicted_k = torch.randn(2, attn.num_kv_heads, attn.head_dim, requires_grad=True)
    evicted_v = torch.randn(2, attn.num_kv_heads, attn.head_dim, requires_grad=True)

    updates = attn._project_memory_update(evicted_k, evicted_v)
    updates.square().sum().backward()

    assert evicted_k.grad is not None
    assert evicted_v.grad is not None
    assert evicted_k.grad.norm() > 0
    assert evicted_v.grad.norm() > 0


def test_mem_scale_remains_learnable():
    model = GatedMemSWAForCausalLM(build_config(mem_update_source="kv"))
    attn = model.model.layers[0].attn.float()

    assert isinstance(attn.log_mem_scale, nn.Parameter)

    evicted_k = torch.randn(2, attn.num_kv_heads, attn.head_dim, requires_grad=True)
    evicted_v = torch.randn(2, attn.num_kv_heads, attn.head_dim, requires_grad=True)
    updates = attn._project_memory_update(evicted_k, evicted_v)
    loss = (updates.square().sum() * attn.mem_scale.float().sum())
    loss.backward()

    assert attn.log_mem_scale.grad is not None
    assert torch.isfinite(attn.log_mem_scale.grad).all()


@torch.inference_mode(False)
def test_varlen_kv_memory_cuda_training_step_stays_finite():
    if not torch.cuda.is_available() or flash_attn_varlen_func is None:
        return

    torch.manual_seed(0)
    model = GatedMemSWAForCausalLM(
        build_config(
            hidden_size=96,
            num_hidden_layers=2,
            num_heads=6,
            num_kv_heads=3,
            intermediate_size=192,
            window_size=16,
            max_position_embeddings=512,
            mem_update_source="kv",
        )
    ).to("cuda", dtype=torch.bfloat16)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-15)

    batch = torch.randint(0, model.config.vocab_size, (1, 256), device="cuda")
    cu_seqlens = torch.tensor([0, 53, 117, 181, 256], device="cuda", dtype=torch.int32)
    position_ids = torch.cat(
        [torch.arange(length, device="cuda", dtype=torch.long) for length in (53, 64, 64, 75)]
    ).unsqueeze(0)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=batch,
            labels=batch,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
        )
        assert torch.isfinite(output.loss)
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(grad_norm)
        optimizer.step()


@torch.inference_mode(False)
def test_cuda_memory_scan_preserves_gradients():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    model = GatedMemSWAForCausalLM(build_config()).to("cuda", dtype=torch.bfloat16)
    attn = model.model.layers[0].attn

    gates = torch.full(
        (1, 512, attn.num_kv_heads, 1),
        0.731,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    updates = torch.randn(
        1,
        512,
        attn.num_kv_heads,
        attn.num_mem_slots,
        attn.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    memory_state = torch.zeros(
        1,
        attn.num_kv_heads,
        attn.num_mem_slots,
        attn.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    state_seq, _ = attn._run_shared_gate_memory_scan(
        gates,
        updates,
        memory_state,
        output_final_state=False,
    )
    loss = state_seq.float().square().mean()
    loss.backward()

    assert state_seq.requires_grad
    assert gates.grad is not None and torch.isfinite(gates.grad).all()
    assert updates.grad is not None and torch.isfinite(updates.grad).all()
