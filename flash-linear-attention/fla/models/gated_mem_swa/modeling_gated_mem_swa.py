from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.gated_mem_swa import GatedMemSWA
from fla.models.gated_mem_swa.configuration_gated_mem_swa import GatedMemSWAConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as GatedMemSWAMLP
from fla.modules.l2warp import l2_warp

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


class GatedMemSWABlock(GradientCheckpointingLayer):

    def __init__(self, config: GatedMemSWAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedMemSWA(
            dim=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            disable_memory=config.disable_memory,
            disable_local=getattr(config, "disable_local", False),
            mem_separate_proj=config.mem_separate_proj,
            mem_evicted_only=config.mem_evicted_only,
            mem_use_short_conv=getattr(config, "mem_use_short_conv", False),
            mem_conv_size=getattr(config, "mem_conv_size", 4),
            mem_use_output_norm=getattr(config, "mem_use_output_norm", False),
            mem_swa_drop_prob=getattr(config, "mem_swa_drop_prob", 0.0),
            mem_gate_logit_bias=config.mem_gate_logit_bias,
            mix_gate_logit_bias=config.mix_gate_logit_bias,
            a_log_init_lo=config.a_log_init_lo,
            a_log_init_hi=config.a_log_init_hi,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            layer_idx=layer_idx,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMemSWAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)
        return outputs


class GatedMemSWAPreTrainedModel(PreTrainedModel):

    config_class = GatedMemSWAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GatedMemSWABlock"]
    _supports_cache_class = True
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        # GM-SWA memory params (A_log / dt_bias / fused gate bias) must be
        # initialized with whole-tensor ops so they survive FSDP's DTensor
        # sharding. `reset_parameters` (view + row-select) only works on plain
        # tensors and is kept for the non-distributed path (layer __init__).
        if isinstance(module, GatedMemSWA) and next(module.parameters()).device.type != "meta":
            if module.memory_enabled:
                self._init_gmswa_memory_params(module)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)

    @torch.no_grad()
    def _init_gmswa_memory_params(self, module: "GatedMemSWA") -> None:
        """DTensor-safe init for the v2 memory branch (A_log / dt_bias / fused gate bias).

        Uses only whole-tensor ops (uniform_/clamp/log/copy_) and `distribute_tensor`
        so it works whether the params are plain tensors (tests) or FSDP DTensors.
        """
        H = module.num_heads
        # Fused gate projection: weight ~ N(0, 0.02); bias laid out as
        # [beta(=mem_gate_bias) | a(=0) | mix(=mix_gate_bias)] over 3*H entries.
        nn.init.normal_(module.gate_proj.weight, mean=0.0, std=0.02)
        bias = module.gate_proj.bias
        full = torch.empty(3 * H, dtype=bias.dtype, device=bias.device)
        full[:H].fill_(module.mem_gate_logit_bias)
        full[H:2 * H].zero_()
        full[2 * H:].fill_(module.mix_gate_logit_bias)
        try:
            from torch.distributed.tensor import DTensor, distribute_tensor
        except Exception:
            DTensor = None
        if DTensor is not None and isinstance(bias, DTensor):
            bias.copy_(distribute_tensor(full, bias.device_mesh, bias.placements))
        else:
            bias.copy_(full)

        # Mamba-2 style decay params.
        nn.init.uniform_(module.A_log, a=math.log(module.a_log_init_lo), b=math.log(module.a_log_init_hi))
        module.A_log._no_weight_decay = True
        nn.init.uniform_(module.dt_bias, a=module.dt_min, b=module.dt_max)
        dt = module.dt_bias.clamp(min=module.dt_init_floor)
        module.dt_bias.copy_(torch.log(torch.expm1(dt)))  # inverse softplus
        module.dt_bias._no_weight_decay = True


class GatedMemSWAModel(GatedMemSWAPreTrainedModel):

    def __init__(self, config: GatedMemSWAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GatedMemSWABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        # Memory-first curriculum: anneal the SWA-drop prob from its start value to
        # 0 over the first `mem_swa_drop_anneal_steps` training forwards, so the
        # recurrent memory is forced to learn sharp recall alone early, then the
        # hybrid converges normally. Counter lives here (outside the checkpointed
        # block) so activation-checkpoint recompute stays consistent.
        self._mem_drop_p0 = float(getattr(config, "mem_swa_drop_prob", 0.0))
        self._mem_drop_anneal = int(getattr(config, "mem_swa_drop_anneal_steps", 0))
        self._mem_drop_step = 0

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`GatedMemSWAModel` does not support output attention weights now, "
                "so `output_attentions` is set to `False`.",
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        # memory-first curriculum: set this step's annealed SWA-drop prob on each layer
        if self.training and self._mem_drop_anneal > 0:
            eff_p = self._mem_drop_p0 * max(0.0, 1.0 - self._mem_drop_step / self._mem_drop_anneal)
            for _l in self.layers:
                _l.attn._runtime_drop_p = eff_p
            self._mem_drop_step += 1

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class GatedMemSWAForCausalLM(GatedMemSWAPreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GatedMemSWAConfig):
        super().__init__(config)
        self.model = GatedMemSWAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
