from pydantic import BaseModel
from typing import List, Optional

class ClosePositionsRequest(BaseModel):
    symbols: List[str]

class LLMSettingsUpdateRequest(BaseModel):
    provider: str
    grok_model: Optional[str] = None
    local_model: Optional[str] = None
    local_url: Optional[str] = None
    position_monitor_interval_seconds: Optional[int] = 60
    take_profit_percentage: Optional[float] = None
    stop_loss_percentage: Optional[float] = None
    daily_drawdown_threshold: Optional[float] = None
    web_research_enabled: Optional[bool] = None
    web_research_max_tickers: Optional[int] = None
    web_research_days: Optional[int] = None

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
