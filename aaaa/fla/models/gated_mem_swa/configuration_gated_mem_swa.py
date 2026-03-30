
import warnings

from transformers.configuration_utils import PretrainedConfig


class GatedMemSWAConfig(PretrainedConfig):

    model_type = "gated_mem_swa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = True,
        window_size: int = 4096,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        num_mem_slots: int = 1,
        mem_scale: float = 1.0,
        mem_rank: int | None = None,
        mem_proj_mode: str = "linear",
        mem_gate_mode: str = "linear",
        mem_update_stride: int = 1,
        mem_token_threshold: int | None = None,
        disable_memory: bool = False,
        gate_bias_init: float = 1.0,
        mem_norm: bool = True,
        mem_norm_eps: float = 1e-6,
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
        self.num_mem_slots = num_mem_slots
        self.mem_scale = mem_scale
        self.mem_rank = mem_rank
        self.mem_proj_mode = mem_proj_mode
        self.mem_gate_mode = mem_gate_mode
        self.mem_update_stride = mem_update_stride
        self.mem_token_threshold = mem_token_threshold
        self.disable_memory = disable_memory
        self.gate_bias_init = gate_bias_init
        self.mem_norm = mem_norm
        self.mem_norm_eps = mem_norm_eps

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

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
