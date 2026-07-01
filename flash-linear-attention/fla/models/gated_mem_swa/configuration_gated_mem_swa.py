from __future__ import annotations

import warnings

from transformers.configuration_utils import PretrainedConfig


class GatedMemSWAConfig(PretrainedConfig):
    """Configuration for the GM-SWA v2 model (paper/gmswa_v2_design.md)."""

    model_type = "gated_mem_swa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        window_size: int = 512,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        # ---- optional hybrid Transformer layers ----
        attn: dict | None = None,
        # ---- v2 memory branch ----
        disable_memory: bool = False,
        disable_local: bool = False,
        mem_separate_proj: bool = False,
        mem_evicted_only: bool = True,
        mem_use_short_conv: bool = False,
        mem_conv_size: int = 4,
        mem_use_output_norm: bool = False,
        mem_swa_drop_prob: float = 0.0,
        mem_swa_drop_anneal_steps: int = 0,
        mem_gate_logit_bias: float = -2.0,
        mix_gate_logit_bias: float = 4.0,
        a_log_init_lo: float = 1.0,
        a_log_init_hi: float = 16.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        # ---- MLP / norm / fuse ----
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attn = attn

        self.disable_memory = disable_memory
        self.disable_local = disable_local
        self.mem_separate_proj = mem_separate_proj
        self.mem_evicted_only = mem_evicted_only
        self.mem_use_short_conv = mem_use_short_conv
        self.mem_conv_size = mem_conv_size
        self.mem_use_output_norm = mem_use_output_norm
        self.mem_swa_drop_prob = mem_swa_drop_prob
        self.mem_swa_drop_anneal_steps = mem_swa_drop_anneal_steps
        self.mem_gate_logit_bias = mem_gate_logit_bias
        self.mix_gate_logit_bias = mix_gate_logit_bias
        self.a_log_init_lo = a_log_init_lo
        self.a_log_init_hi = a_log_init_hi
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        # Silently ignore obsolete v1 fields so old checkpoints still load.
        _legacy = (
            "num_mem_slots", "num_memory_components", "use_memory_component",
            "memory_state_rank", "mem_scale", "mem_rank", "mem_proj_mode",
            "mem_gate_mode", "mem_update_source", "mem_update_stride",
            "mem_token_threshold", "gate_bias_init", "mem_norm", "mem_norm_eps",
        )
        legacy_present = [k for k in _legacy if k in kwargs]
        if legacy_present:
            warnings.warn(
                "Ignoring obsolete v1 GM-SWA config fields: "
                f"{', '.join(legacy_present)}. See paper/gmswa_v2_design.md."
            )
            for k in legacy_present:
                kwargs.pop(k, None)

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if "layers" not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if "num_heads" not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn["num_kv_heads"] = attn.get("num_kv_heads", attn["num_heads"])
            attn["qkv_bias"] = attn.get("qkv_bias", False)
            attn["qk_norm"] = attn.get("qk_norm", False)
            attn["window_size"] = attn.get("window_size", None)
            attn["rope_theta"] = attn.get("rope_theta", self.rope_theta)

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time."
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
