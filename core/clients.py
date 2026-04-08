from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from core.config import settings

# Initialize Clients
trading_client = None
data_client = None
grok_client = None

if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY:
    trading_client = TradingClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, paper=settings.ALPACA_PAPER)
    data_client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)

if settings.XAI_API_KEY:
    grok_client = AsyncOpenAI(
        api_key=settings.XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )
elif settings.OPENROUTER_API_KEY:
    grok_client = AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

# Bot Status
bot_state = {
    "BOT_ACTIVE": True
}
