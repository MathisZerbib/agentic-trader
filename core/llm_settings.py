import json
import os
from core.config import settings

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".data", "llm_settings.json")

# Default settings
LLM_SETTINGS = {
    "provider": settings.LLM_PROVIDER if hasattr(settings, "LLM_PROVIDER") else "grok",
    "grok_model": settings.DEFAULT_GROK_MODEL,
    "local_model": "local-model",
    "local_url": settings.LOCAL_LLM_URL,
    "position_monitor_interval_seconds": settings.POSITION_MONITOR_INTERVAL_SECONDS,
    "take_profit_percentage": settings.TAKE_PROFIT_PERCENTAGE,
    "stop_loss_percentage": settings.STOP_LOSS_PERCENTAGE,
    "daily_drawdown_threshold": settings.DAILY_DRAWDOWN_THRESHOLD,
    "web_research_enabled": settings.WEB_RESEARCH_ENABLED,
    "web_research_max_tickers": settings.WEB_RESEARCH_MAX_TICKERS,
    "web_research_days": settings.WEB_RESEARCH_DAYS,
}

# Load from disk if exists
if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
            saved = json.load(f)
            LLM_SETTINGS.update(saved)
    except Exception as e:
        print(f"Error loading LLM settings: {e}")
