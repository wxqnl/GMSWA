from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch


@dataclass
class MemoryTraceRecorder:
    records: list[dict[str, Any]] = field(default_factory=list)
    current_sample_id: str | None = None
    step_counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _patched_modules: list[tuple[object, str, Any]] = field(default_factory=list)

    def set_sample(self, sample_id: str | None) -> None:
        if sample_id != self.current_sample_id:
            self.step_counters.clear()
        self.current_sample_id = sample_id

    def attach(self, model: torch.nn.Module) -> None:
        for module_name, module in model.named_modules():
            if module.__class__.__name__ != "GatedMemSWA":
                continue
            original_update = module._update_memory
            original_forward_fast = module._forward_fast

            def wrapped_update(this, memory_state, evicted_v, gate_input, *, _module_name=module_name, _orig=original_update):
                new_state = _orig(memory_state, evicted_v, gate_input)
                gate = this._compute_gate(gate_input).detach()
                if gate.dim() == 1:
                    gate = gate.unsqueeze(0)
                slot_norm = new_state.detach().norm(dim=-1)
                batch_index = 0
                gate_values = gate[batch_index].reshape(-1).tolist()
                slot_norm_values = slot_norm[batch_index].reshape(-1).tolist()
                step_id = self.step_counters[_module_name]
                for slot_id, gate_value in enumerate(gate_values):
                    self.records.append(
                        {
                            "sample_id": self.current_sample_id,
                            "step_id": step_id,
                            "slot_id": slot_id,
                            "gate_value": float(gate_value),
                            "slot_norm": float(slot_norm_values[slot_id]),
                            "read_weight": None,
                            "event_type": "write",
                            "event_just_evicted": True,
                            "layer_name": _module_name,
                        }
                    )
                self.step_counters[_module_name] += 1
                return new_state

            def wrapped_forward_fast(this, hidden_states, attention_mask=None, *, seqlen_offset=0, memory_state=None, _module_name=module_name, _orig=original_forward_fast):
                outputs = _orig(
                    hidden_states,
                    attention_mask=attention_mask,
                    seqlen_offset=seqlen_offset,
                    memory_state=memory_state,
                )
                if this.disable_memory:
                    return outputs

                batch_size, seq_len, _ = hidden_states.shape
                if seq_len <= this.window_size:
                    return outputs

                q = this.q_proj(hidden_states).view(batch_size, seq_len, this.num_heads, this.head_dim)
                k = this.k_proj(hidden_states).view(batch_size, seq_len, this.num_kv_heads, this.head_dim)
                v = this.v_proj(hidden_states).view(batch_size, seq_len, this.num_kv_heads, this.head_dim)
                q, k = this._apply_rope(q, k, seqlen_offset=seqlen_offset)
                if hasattr(this, "num_kv_groups") and this.num_kv_groups > 1:
                    from fla12131231.layers.gated_mem_swa import repeat_kv

                    k = repeat_kv(k, this.num_kv_groups)
                    v = repeat_kv(v, this.num_kv_groups)

                if memory_state is None:
                    memory_state = torch.zeros(
                        (batch_size, this.num_heads, this.head_dim),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )

                gate = this._compute_gate(hidden_states).unsqueeze(-1)
                evicted_v = v[:, : seq_len - this.window_size]
                update = this._project_memory_update(evicted_v)
                pad = torch.zeros(
                    (batch_size, this.window_size, this.num_heads, this.head_dim),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                update = torch.cat([pad, update], dim=1)

                positions = torch.arange(seq_len, device=hidden_states.device)
                update_mask = positions >= this.window_size
                if this.mem_update_stride > 1:
                    update_mask &= ((seqlen_offset + positions - this.window_size) % this.mem_update_stride) == 0
                update_mask = update_mask.view(1, seq_len, 1, 1)

                gate = torch.where(update_mask, gate, torch.ones_like(gate))
                update = torch.where(update_mask, update, torch.zeros_like(update))

                eps = this.mem_norm_eps
                gate_fp32 = gate.squeeze(-1).to(torch.float32)
                log_gate = torch.log(torch.clamp(gate_fp32, min=eps))
                log_prefix = torch.cumsum(log_gate, dim=1)
                prefix = torch.exp(log_prefix).clamp_min(eps).unsqueeze(-1)
                inv_prefix = 1.0 / prefix
                contrib = ((1.0 - gate.to(torch.float32)) * update.to(torch.float32)) * inv_prefix
                accum = torch.cumsum(contrib, dim=1)
                memory_seq = prefix * (memory_state.to(torch.float32).unsqueeze(1) + accum)

                batch_index = 0
                base_step = 0
                for t in range(seq_len):
                    if not bool(update_mask[0, t, 0, 0].item()):
                        continue
                    gate_values = gate[batch_index, t].squeeze(-1).detach().reshape(-1).tolist()
                    slot_norm_values = memory_seq[batch_index, t].detach().norm(dim=-1).reshape(-1).tolist()
                    step_id = self.step_counters[_module_name] + base_step
                    for slot_id, gate_value in enumerate(gate_values):
                        self.records.append(
                            {
                                "sample_id": self.current_sample_id,
                                "step_id": step_id,
                                "slot_id": slot_id,
                                "gate_value": float(gate_value),
                                "slot_norm": float(slot_norm_values[slot_id]),
                                "read_weight": None,
                                "event_type": "write",
                                "event_just_evicted": True,
                                "layer_name": _module_name,
                            }
                        )
                    base_step += 1
                self.step_counters[_module_name] += base_step
                return outputs

            module._update_memory = MethodType(wrapped_update, module)
            module._forward_fast = MethodType(wrapped_forward_fast, module)
            self._patched_modules.append((module, "_update_memory", original_update))
            self._patched_modules.append((module, "_forward_fast", original_forward_fast))

    def detach(self) -> None:
        for module, attr_name, original in self._patched_modules:
            setattr(module, attr_name, original)
        self._patched_modules.clear()
