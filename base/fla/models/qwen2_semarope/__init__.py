from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from fla.models.qwen2_semarope.configuration_qwen2_semarope import Qwen2SemaRoPEConfig
from fla.models.qwen2_semarope.modeling_qwen2_semarope import (
    Qwen2SemaRoPEForCausalLM,
    Qwen2SemaRoPEForQuestionAnswering,
    Qwen2SemaRoPEForSequenceClassification,
    Qwen2SemaRoPEForTokenClassification,
    Qwen2SemaRoPEModel,
    Qwen2SemaRoPEPreTrainedModel,
)
AutoConfig.register(Qwen2SemaRoPEConfig.model_type, Qwen2SemaRoPEConfig, exist_ok=True)
AutoModel.register(Qwen2SemaRoPEConfig, Qwen2SemaRoPEModel, exist_ok=True)
AutoModelForCausalLM.register(Qwen2SemaRoPEConfig, Qwen2SemaRoPEForCausalLM, exist_ok=True)


__all__ = [
    "Qwen2SemaRoPEConfig",
    "Qwen2SemaRoPEForCausalLM",
    "Qwen2SemaRoPEForQuestionAnswering",
    "Qwen2SemaRoPEForSequenceClassification",
    "Qwen2SemaRoPEForTokenClassification",
    "Qwen2SemaRoPEModel",
    "Qwen2SemaRoPEPreTrainedModel",
]
