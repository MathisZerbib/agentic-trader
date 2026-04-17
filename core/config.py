import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "grok")
    LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://host.docker.internal:1234/v1")
    LOCAL_LLM_MAX_CHARS = int(os.getenv("LOCAL_LLM_MAX_CHARS", "64000"))
    WEB_RESEARCH_ENABLED = os.getenv("WEB_RESEARCH_ENABLED", "true").lower() == "true"
    WEB_RESEARCH_MAX_TICKERS = int(os.getenv("WEB_RESEARCH_MAX_TICKERS", "3"))
    WEB_RESEARCH_MACRO_MAX_RESULTS = int(os.getenv("WEB_RESEARCH_MACRO_MAX_RESULTS", "6"))
    WEB_RESEARCH_TICKER_MAX_RESULTS = int(os.getenv("WEB_RESEARCH_TICKER_MAX_RESULTS", "5"))
    WEB_RESEARCH_DAYS = int(os.getenv("WEB_RESEARCH_DAYS", "3"))
    
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_PAPER = os.getenv("ALPACA_PAPER", "True").lower() == "true"
    
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "x-ai/grok-4.1-fast")
    LM_STUDIO_API_TOKEN = os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY")
    
    TAKE_PROFIT_PERCENTAGE = 0.05
    STOP_LOSS_PERCENTAGE = -0.03
    DAILY_DRAWDOWN_THRESHOLD = float(os.getenv("DAILY_DRAWDOWN_THRESHOLD", "-0.03"))
    POSITION_MONITOR_INTERVAL_SECONDS = int(os.getenv("POSITION_MONITOR_INTERVAL_SEC", "60"))

settings = Settings()
