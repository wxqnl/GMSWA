import torch
import torch.nn as nn

import fla.layers.gated_mem_swa as gated_mem_swa_module
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
    expected_offsets = torch.tensor([1.0], dtype=gates.dtype)
    expected = torch.sigmoid(expected_offsets).view(1, 1, 1, -1).expand_as(gates)
    assert torch.allclose(gates, expected, atol=1e-6, rtol=0)


def test_memory_component_alias_keeps_single_component_semantics():
    model = GatedMemSWAForCausalLM(build_config(num_memory_components=1))
    attn = model.model.layers[0].attn

    assert model.config.num_memory_components == 1
    assert model.config.num_mem_slots == 1
    assert attn.num_memory_components == 1
    assert attn.num_mem_slots == 1


def test_use_memory_component_flag_controls_memory_path():
    enabled = GatedMemSWAForCausalLM(build_config(use_memory_component=True))
    disabled = GatedMemSWAForCausalLM(build_config(use_memory_component=False))

    assert enabled.config.use_memory_component is True
    assert enabled.model.layers[0].attn.memory_enabled is True
    assert disabled.config.use_memory_component is False
    assert disabled.model.layers[0].attn.memory_enabled is False
    assert disabled.model.layers[0].attn.num_memory_components == 0


def test_projected_memory_updates_are_bounded():
    model = GatedMemSWAForCausalLM(build_config(num_mem_slots=4, mem_update_source="value"))
    attn = model.model.layers[0].attn
    evicted_v = torch.randn(2, 16, attn.num_kv_heads, attn.head_dim) * 10

    updates = attn._project_memory_update(None, evicted_v)
    rms = updates.float().square().mean(dim=-1).sqrt()

    assert torch.all(rms <= 1.0001)


def test_recurrent_memory_state_stays_bounded():
    model = GatedMemSWAForCausalLM(build_config(num_mem_slots=4, mem_update_source="value"))
    attn = model.model.layers[0].attn
    memory_state = torch.zeros(2, attn.num_kv_heads, attn.num_mem_slots, attn.head_dim)

    for _ in range(64):
        evicted_v = torch.randn(2, attn.num_kv_heads, attn.head_dim) * 10
        memory_state = attn._update_memory(memory_state, None, evicted_v)
        rms = memory_state.float().square().mean(dim=-1).sqrt()
        assert torch.all(rms <= 1.0001)


def test_memory_clip_is_head_dim_invariant():
    model = GatedMemSWAForCausalLM(build_config(mem_update_source="value"))
    attn = model.model.layers[0].attn.float()

    # Unit-RMS states should not be shrunk just because they live in a
    # 64-dimensional head.
    state = torch.ones(2, attn.num_kv_heads, attn.memory_state_rows, attn.head_dim)
    normalized = attn._normalize_memory(state)

    assert torch.allclose(normalized, state)


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


def test_single_slot_value_readout_preserves_memory_magnitude():
    model = GatedMemSWAForCausalLM(build_config(mem_update_source="kv"))
    attn = model.model.layers[0].attn.float()

    q_t = torch.ones(1, attn.num_heads, attn.head_dim)
    o_local = torch.zeros(1, attn.num_heads, attn.head_dim)
    lse_local = torch.full((1, attn.num_heads), -100.0)

    base_state = torch.randn(1, attn.num_kv_heads, attn.memory_state_rows, attn.head_dim)
    small_state = base_state * 0.25
    large_state = base_state * 1.0

    small_out = attn._combine_single_memory_step(q_t, o_local, lse_local, small_state)
    large_out = attn._combine_single_memory_step(q_t, o_local, lse_local, large_state)

    assert large_out.norm() > small_out.norm()


def test_single_component_read_is_parallel_to_local_lse():
    model = GatedMemSWAForCausalLM(build_config(mem_update_source="kv"))
    attn = model.model.layers[0].attn.float()

    q_t = torch.ones(1, attn.num_heads, attn.head_dim)
    o_local = torch.zeros(1, attn.num_heads, attn.head_dim)
    memory_state = torch.zeros(1, attn.num_kv_heads, attn.memory_state_rows, attn.head_dim)
    memory_state[:, :, 0] = 1.0
    memory_state[:, :, 1] = 1.0

    low_lse = torch.zeros(1, attn.num_heads)
    high_lse = torch.full((1, attn.num_heads), 8.0)

    low_out = attn._combine_single_memory_step(q_t, o_local, low_lse, memory_state)
    high_out = attn._combine_single_memory_step(q_t, o_local, high_lse, memory_state)

    assert torch.allclose(low_out, high_out)


def test_single_component_writes_innovation_not_raw_value():
    model = GatedMemSWAForCausalLM(build_config(mem_update_source="value"))
    attn = model.model.layers[0].attn.float()

    evicted_v = torch.randn(2, attn.num_kv_heads, attn.head_dim)
    matched_local = evicted_v.clone()
    unmatched_local = torch.zeros_like(evicted_v)

    matched_update = attn._project_memory_update(None, evicted_v, local_summary=matched_local)
    unmatched_update = attn._project_memory_update(None, evicted_v, local_summary=unmatched_local)

    assert matched_update.norm() < unmatched_update.norm()


def test_write_gate_depends_on_evicted_content_not_current_hidden():
    model = GatedMemSWAForCausalLM(build_config(mem_update_source="value"))
    attn = model.model.layers[0].attn.float()

    with torch.no_grad():
        attn.gate_net.weight.zero_()
        attn.gate_net.bias.zero_()
        attn.gate_net.weight[:, 0] = 1.0

    hidden_a = torch.randn(1, 4, attn.dim)
    hidden_b = torch.randn(1, 4, attn.dim) * 100
    k = torch.zeros(1, 4, attn.num_kv_heads, attn.head_dim)
    v = torch.zeros(1, 4, attn.num_kv_heads, attn.head_dim)
    v[:, 0, 0, 0] = 2.0

    gates_a, _ = attn._build_memory_inputs(hidden_a, k, v)
    gates_b, _ = attn._build_memory_inputs(hidden_b, k, v)

    assert torch.allclose(gates_a, gates_b)
    assert gates_a[0, 0, 0, 0] > 0.5


def test_single_component_waits_for_window_eviction_before_updates():
    model = GatedMemSWAForCausalLM(build_config(window_size=8, mem_update_source="value"))
    attn = model.model.layers[0].attn.float()

    hidden = torch.randn(1, 4, attn.dim)
    k = torch.randn(1, 4, attn.num_kv_heads, attn.head_dim)
    v = torch.randn(1, 4, attn.num_kv_heads, attn.head_dim)

    _, updates = attn._build_memory_inputs(hidden, k, v)

    assert updates.abs().sum() == 0


def test_single_component_uses_a_single_write_gate():
    model = GatedMemSWAForCausalLM(build_config(window_size=2, mem_update_source="value"))
    attn = model.model.layers[0].attn.float()

    gates = attn._compute_gate(torch.zeros(1, 1, attn.dim)).squeeze(0).squeeze(0)

    assert gates.shape[-1] == 1


def test_legacy_cache_state_still_recovers_local_summary():
    model = GatedMemSWAForCausalLM(build_config(mem_update_source="value"))
    attn = model.model.layers[0].attn.float()
    batch_size = 1
    cache_len = 3

    k_cached = torch.randn(batch_size, cache_len, attn.num_heads, attn.head_dim)
    v_cached = torch.randn(batch_size, cache_len, attn.num_heads, attn.head_dim)
    k_write_cached = attn._collapse_query_groups(k_cached)
    legacy_state = [
        {
            "attn_state": (
                k_cached.reshape(batch_size, cache_len, -1),
                v_cached.reshape(batch_size, cache_len, -1),
                k_write_cached.reshape(batch_size, cache_len, -1),
            )
        }
    ]

    _, _, _, _, local_summary = attn._load_cached_state(
        legacy_state,
        batch_size=batch_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert local_summary is not None
    assert local_summary.shape == (batch_size, cache_len, attn.num_kv_heads, attn.head_dim)


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
        attn.memory_state_rows,
        attn.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    memory_state = torch.zeros(
        1,
        attn.num_kv_heads,
        attn.memory_state_rows,
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


@torch.inference_mode(False)
def test_cuda_training_path_keeps_full_memory_credit_assignment():
    if not torch.cuda.is_available() or flash_attn_varlen_func is None:
        return

    torch.manual_seed(0)
    model = GatedMemSWAForCausalLM(
        build_config(
            hidden_size=96,
            num_hidden_layers=1,
            num_heads=6,
            num_kv_heads=3,
            intermediate_size=192,
            window_size=8,
            max_position_embeddings=128,
            mem_update_source="kv",
        )
    ).to("cuda", dtype=torch.bfloat16)
    model.train()

    original = gated_mem_swa_module.fused_recurrent_gm_swa
    truncate_flags = []

    def wrapped(*args, **kwargs):
        truncate_flags.append(kwargs.get("truncate_backward"))
        return original(*args, **kwargs)

    gated_mem_swa_module.fused_recurrent_gm_swa = wrapped
    try:
        batch = torch.randint(0, model.config.vocab_size, (1, 64), device="cuda")
        cu_seqlens = torch.tensor([0, 19, 41, 64], device="cuda", dtype=torch.int32)
        position_ids = torch.cat(
            [torch.arange(length, device="cuda", dtype=torch.long) for length in (19, 22, 23)]
        ).unsqueeze(0)
        loss = model(
            input_ids=batch,
            labels=batch,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
        ).loss
        loss.backward()
    finally:
        gated_mem_swa_module.fused_recurrent_gm_swa = original

    assert truncate_flags
    assert all(flag is False for flag in truncate_flags)
