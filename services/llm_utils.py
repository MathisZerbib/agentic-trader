import os
import json
import urllib.request
import requests
from core.config import settings

def get_active_local_model_sync():
    """
    Synchronous version of fetching currently loaded model(s) from LM Studio.
    """
    local_url = os.getenv("LOCAL_LLM_URL", "http://host.docker.internal:1234/v1")
    if "/chat/completions" in local_url:
        base_url = local_url.replace("/chat/completions", "")
    else:
        base_url = local_url.rstrip("/")

    models_url = f"{base_url}/models"
    
    try:
        headers = {}
        if settings.LM_STUDIO_API_TOKEN:
            headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_TOKEN}"
            
        response = requests.get(models_url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        models_data = data.get("data", [])
        
        if models_data:
            return models_data[0].get("id")
    except Exception as e:
        print(f"Failed to autodetect local model (sync): {e}")
    
    return os.getenv("LOCAL_LLM_MODEL", "local-model")

async def get_active_local_model_async():
    """
    Asynchronous version of fetching currently loaded model(s) from LM Studio.
    """
    import asyncio
    return await asyncio.to_thread(get_active_local_model_sync)
