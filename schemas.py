from pydantic import BaseModel
from typing import List, Optional

class ClosePositionsRequest(BaseModel):
    symbols: List[str]

class LLMSettingsUpdateRequest(BaseModel):
    provider: str
    grok_model: Optional[str] = None
    local_model: Optional[str] = None
    local_url: Optional[str] = None
