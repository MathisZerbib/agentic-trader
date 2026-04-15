from pydantic import BaseModel
from typing import List, Optional

class ClosePositionsRequest(BaseModel):
    symbols: List[str]

class LLMSettingsUpdateRequest(BaseModel):
    provider: str
    grok_model: Optional[str] = None
    local_model: Optional[str] = None
    local_url: Optional[str] = None

class LoadModelRequest(BaseModel):
    model: str
    context_length: Optional[int] = None
    eval_batch_size: Optional[int] = None
    flash_attention: Optional[bool] = None
    num_experts: Optional[int] = None
    offload_kv_cache_to_gpu: Optional[bool] = None
    echo_load_config: Optional[bool] = False

class ModelUnloadRequest(BaseModel):
    instance_id: str
