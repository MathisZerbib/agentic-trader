import json
import os
from core.config import settings

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".data", "llm_settings.json")

# Default settings
LLM_SETTINGS = {
    "provider": "local" if settings.LOCAL_LLM_MODEL else "grok",
    "grok_model": settings.DEFAULT_GROK_MODEL,
    "local_model": settings.LOCAL_LLM_MODEL,
    "local_url": settings.LOCAL_LLM_URL,
    "position_monitor_interval_seconds": settings.POSITION_MONITOR_INTERVAL_SECONDS,
}

# Load from disk if exists
if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
            saved = json.load(f)
            LLM_SETTINGS.update(saved)
    except Exception as e:
        print(f"Error loading LLM settings: {e}")
