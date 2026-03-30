
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla12131231.models.gated_mem_swa.configuration_gated_mem_swa import GatedMemSWAConfig
from fla12131231.models.gated_mem_swa.modeling_gated_mem_swa import GatedMemSWAForCausalLM, GatedMemSWAModel

AutoConfig.register(GatedMemSWAConfig.model_type, GatedMemSWAConfig, exist_ok=True)
AutoModel.register(GatedMemSWAConfig, GatedMemSWAModel, exist_ok=True)
AutoModelForCausalLM.register(GatedMemSWAConfig, GatedMemSWAForCausalLM, exist_ok=True)

__all__ = ["GatedMemSWAConfig", "GatedMemSWAForCausalLM", "GatedMemSWAModel"]
