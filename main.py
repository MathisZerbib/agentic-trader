# --- API: Close/Sell Selected Positions ---
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import datetime
import asyncio
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetPortfolioHistoryRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.enums import DataFeed
# Market Sell
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from openai import AsyncOpenAI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import models
import agents
from database import SessionLocal, engine
from ai_trader import (
    format_news_for_prompt,
    format_sentiment_for_prompt,
    format_tavily_for_prompt,
    get_active_stocks,
    get_latest_news,
    get_macro_web_research,
    get_social_sentiment,
    get_ticker_web_research,
    shortlist_candidates_for_web_research,
)
import agent_prompts

from starlette.websockets import WebSocketState

import csv
import io
import urllib.request
import agents # Ensure agents is imported at the start of the function scope


load_dotenv()
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "local-model")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://host.docker.internal:1234/v1")

# Trading Configuration
TAKE_PROFIT_PERCENTAGE = 0.05  # 5%
STOP_LOSS_PERCENTAGE = -0.03   # -3%
WEB_RESEARCH_ENABLED = os.getenv("WEB_RESEARCH_ENABLED", "true").lower() == "true"
WEB_RESEARCH_MAX_TICKERS = int(os.getenv("WEB_RESEARCH_MAX_TICKERS", "3"))
WEB_RESEARCH_MACRO_MAX_RESULTS = int(os.getenv("WEB_RESEARCH_MACRO_MAX_RESULTS", "6"))
WEB_RESEARCH_TICKER_MAX_RESULTS = int(os.getenv("WEB_RESEARCH_TICKER_MAX_RESULTS", "5"))
WEB_RESEARCH_DAYS = int(os.getenv("WEB_RESEARCH_DAYS", "3"))

models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="Grok Trading Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send initial data
        db = SessionLocal()
        try:
            # Send initial state
            state = get_current_state(db)
            await websocket.send_text(json.dumps(state))
            
            # Send initial logs
            logs = db.query(models.AgentLog).order_by(models.AgentLog.timestamp.desc()).limit(50).all()
            logs_msg = {
                "type": "logs",
                "data": [{"timestamp": str(log.timestamp), "title": log.title, "content": log.content} for log in logs]
            }
            await websocket.send_text(json.dumps(logs_msg))
        except Exception as e:
            print(f"Error sending initial WebSocket data: {e}")
            self.disconnect(websocket)
            try:
                await websocket.close()
            except Exception:
                pass
        finally:
            db.close()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        data = json.dumps(message)

        connections = list(self.active_connections)

        async def _send_one(connection: WebSocket):
            if connection.application_state != WebSocketState.CONNECTED:
                raise RuntimeError("WebSocket not connected")
            await asyncio.wait_for(connection.send_text(data), timeout=1.0)

        results = await asyncio.gather(
            *(_send_one(connection) for connection in connections),
            return_exceptions=True,
        )

        for connection, result in zip(connections, results):
            if result is not None:
                self.disconnect(connection)
                try:
                    await connection.close()
                except Exception:
                    pass

manager = ConnectionManager()

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except RuntimeError as e:
        if "WebSocket is not connected" in str(e):
            manager.disconnect(websocket)
        else:
            raise

# Helper to broadcast updates
async def broadcast_ws_message(message: dict):
    await manager.broadcast(message)

def get_current_state(db: Session):
    market_status = "unknown"
    next_open = None
    next_close = None
    qqq_change = 0.0
    
    if trading_client:
        try:
            clock = trading_client.get_clock()
            market_status = "open" if clock.is_open else "closed"
            next_open = str(clock.next_open)
            next_close = str(clock.next_close)
        except Exception as e:
            print(f"Error fetching clock: {e}")

    if data_client:
        try:
            req = StockSnapshotRequest(symbol_or_symbols=["QQQ"])
            snapshot = data_client.get_stock_snapshot(req)
            if "QQQ" in snapshot:
                prev_close = snapshot["QQQ"].previous_daily_bar.close
                latest_trade = snapshot["QQQ"].latest_trade.price
                if prev_close and latest_trade:
                    qqq_change = ((latest_trade - prev_close) / prev_close) * 100
        except Exception as e:
            print(f"Error fetching QQQ data: {e}")

    portfolio_data = None
    if trading_client:
        try:
            account = trading_client.get_account()
            positions = trading_client.get_all_positions()
            portfolio_data = {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "positions": [
                    {
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "market_value": float(p.market_value),
                        "unrealized_pl": float(p.unrealized_pl),
                        "current_price": float(p.current_price) if hasattr(p, 'current_price') else 0.0,
                        "change_today": float(p.unrealized_intraday_plpc) * 100 if hasattr(p, 'unrealized_intraday_plpc') else 0.0
                    } for p in positions
                ]
            }
        except:
            pass

    return {
        "type": "state",
        "bot_active": BOT_ACTIVE,
        "market_status": market_status,
        "next_open": next_open,
        "next_close": next_close,
        "qqq_change": round(qqq_change, 2),
        "portfolio": portfolio_data
    }

async def trigger_state_broadcast():
    db = SessionLocal()
    try:
        state = get_current_state(db)
        await broadcast_ws_message(state)
    finally:
        db.close()

# Background broadcast for the scheduler
async def scheduled_broadcast():
    await trigger_state_broadcast()


# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "True").lower() == "true"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "x-ai/grok-4.1-fast")
LM_STUDIO_API_TOKEN = os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY")

# Initialize Clients
trading_client = None
data_client = None
grok_client = None

LLM_SETTINGS = {
    "provider": os.getenv("LLM_PROVIDER", "local"),
    "grok_model": DEFAULT_GROK_MODEL,
    "local_model": LOCAL_LLM_MODEL,
    "local_url": LOCAL_LLM_URL,
}


def _normalized_local_url(url: str) -> str:
    return url.replace("/chat/completions", "").rstrip("/")


def _sync_local_llm_settings() -> None:
    agents.LOCAL_LLM_MODEL = LLM_SETTINGS["local_model"]
    agents.LOCAL_LLM_URL = _normalized_local_url(LLM_SETTINGS["local_url"])


def _active_engine_label() -> str:
    if agents.USE_LOCAL_FALLBACK:
        return f"local:{agents.LOCAL_LLM_MODEL}"
    return f"grok:{LLM_SETTINGS['grok_model']}"


def _fetch_local_llm_models(local_url: str) -> list[dict]:
    normalized_url = _normalized_local_url(local_url).strip()
    if not normalized_url:
        return []

    candidate_urls = [normalized_url]
    if "localhost" in normalized_url:
        candidate_urls.append(normalized_url.replace("localhost", "host.docker.internal"))
    if "127.0.0.1" in normalized_url:
        candidate_urls.append(normalized_url.replace("127.0.0.1", "host.docker.internal"))

    models_urls = []
    for candidate in list(dict.fromkeys(candidate_urls)):
        root_url = candidate[:-3] if candidate.endswith("/v1") else candidate
        models_urls.append(f"{root_url}/api/v1/models")
        models_urls.append(f"{candidate}/models")
        if not candidate.endswith("/v1"):
            models_urls.append(f"{candidate}/v1/models")

    payload = None
    last_error = None
    for models_url in list(dict.fromkeys(models_urls)):
        try:
            request = urllib.request.Request(models_url)
            if LM_STUDIO_API_TOKEN:
                request.add_header("Authorization", f"Bearer {LM_STUDIO_API_TOKEN}")
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                break
        except Exception as e:
            last_error = e
            continue

    if payload is None:
        print(f"Error fetching LM Studio models from {normalized_url}: {last_error}")
        return []

    raw_models = []
    if isinstance(payload, dict):
        raw_models = payload.get("models") or payload.get("data") or []

    discovered_models = []
    seen_keys = set()
    for entry in raw_models:
        if not isinstance(entry, dict):
            continue

        if entry.get("type") and entry.get("type") != "llm":
            continue

        model_key = (entry.get("key") or entry.get("id") or entry.get("name") or "").strip()
        if not model_key or model_key in seen_keys:
            continue

        seen_keys.add(model_key)
        loaded_instances = entry.get("loaded_instances") or []
        discovered_models.append({
            "key": model_key,
            "display_name": (entry.get("display_name") or model_key).strip(),
            "publisher": entry.get("publisher"),
            "architecture": entry.get("architecture"),
            "max_context_length": entry.get("max_context_length"),
            "format": entry.get("format"),
            "loaded": bool(loaded_instances),
            "loaded_instances": loaded_instances,
            "capabilities": entry.get("capabilities"),
        })

    return discovered_models


agents.USE_LOCAL_FALLBACK = LLM_SETTINGS["provider"] == "local"
_sync_local_llm_settings()

if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

if XAI_API_KEY:
    grok_client = AsyncOpenAI(
        base_url="https://api.x.ai/v1",
        api_key=XAI_API_KEY,
    )
elif OPENROUTER_API_KEY:
    grok_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Scheduler
scheduler = AsyncIOScheduler()
scheduler.add_job(scheduled_broadcast, 'interval', seconds=10)
scheduler.start()

# Bot Status
BOT_ACTIVE = True

class ClosePositionsRequest(BaseModel):
    symbols: list[str]


class LLMSettingsUpdateRequest(BaseModel):
    provider: str
    grok_model: Optional[str] = None
    local_model: Optional[str] = None
    local_url: Optional[str] = None

@app.get("/")
def read_root():
    market_status = "unknown"
    next_open = None
    next_close = None
    qqq_change = 0.0
    
    if trading_client:
        try:
            clock = trading_client.get_clock()
            market_status = "open" if clock.is_open else "closed"
            next_open = clock.next_open
            next_close = clock.next_close
        except Exception as e:
            print(f"Error fetching clock: {e}")

    if data_client:
        try:
            req = StockSnapshotRequest(symbol_or_symbols=["QQQ"])
            snapshot = data_client.get_stock_snapshot(req)
            if "QQQ" in snapshot:
                # Calculate change from previous close
                prev_close = snapshot["QQQ"].previous_daily_bar.close
                latest_trade = snapshot["QQQ"].latest_trade.price
                if prev_close and latest_trade:
                    qqq_change = ((latest_trade - prev_close) / prev_close) * 100
        except Exception as e:
            print(f"Error fetching QQQ data: {e}")

    return {
        "status": "online", 
        "service": "Grok Trading Bot", 
        "bot_active": BOT_ACTIVE,
        "market_status": market_status,
        "next_open": next_open,
        "next_close": next_close,
        "qqq_change": round(qqq_change, 2)
    }

@app.post("/bot/start")
async def start_bot():
    global BOT_ACTIVE
    BOT_ACTIVE = True
    await trigger_state_broadcast()
    return {"status": "Bot started", "bot_active": BOT_ACTIVE}

@app.post("/bot/stop")
async def stop_bot():
    global BOT_ACTIVE
    BOT_ACTIVE = False
    await trigger_state_broadcast()
    return {"status": "Bot stopped", "bot_active": BOT_ACTIVE}


@app.get("/settings/llm")
def get_llm_settings():
    return {
        "provider": LLM_SETTINGS["provider"],
        "grok_model": LLM_SETTINGS["grok_model"],
        "local_model": LLM_SETTINGS["local_model"],
        "local_url": _normalized_local_url(LLM_SETTINGS["local_url"]),
        "recommended_grok_model": DEFAULT_GROK_MODEL,
        "fallback_active": agents.USE_LOCAL_FALLBACK,
        "active_engine": _active_engine_label(),
        "openrouter_available": bool(grok_client),
    }


@app.get("/settings/llm/models")
def get_llm_models(local_url: Optional[str] = None):
    url = local_url.strip() if local_url and local_url.strip() else LLM_SETTINGS["local_url"]
    local_models = _fetch_local_llm_models(url)
    return {
        "local_models": [model["key"] for model in local_models],
        "local_models_detail": local_models,
        "recommended_grok_model": DEFAULT_GROK_MODEL,
        "local_url": _normalized_local_url(url),
    }


@app.post("/settings/llm")
def update_llm_settings(payload: LLMSettingsUpdateRequest):
    provider = payload.provider.lower().strip()
    if provider not in {"grok", "local"}:
        raise HTTPException(status_code=400, detail="provider must be 'grok' or 'local'")

    if provider == "grok" and not grok_client:
        raise HTTPException(status_code=400, detail="Grok provider is not configured (set XAI_API_KEY or OPENROUTER_API_KEY)")

    if payload.grok_model and payload.grok_model.strip():
        LLM_SETTINGS["grok_model"] = payload.grok_model.strip()

    if payload.local_model and payload.local_model.strip():
        LLM_SETTINGS["local_model"] = payload.local_model.strip()

    if payload.local_url and payload.local_url.strip():
        LLM_SETTINGS["local_url"] = payload.local_url.strip()

    LLM_SETTINGS["provider"] = provider
    agents.USE_LOCAL_FALLBACK = provider == "local"
    _sync_local_llm_settings()

    return {
        "status": "LLM settings updated",
        **get_llm_settings(),
    }

@app.get("/portfolio")
def get_portfolio():
    if not trading_client:
        raise HTTPException(status_code=503, detail="Alpaca client not initialized")
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()
    
    # Fetch initial capital (first equity point from history)
    initial_capital = float(account.equity)
    try:
        # We use a broad timeframe to minimize data points, just need the start
        req = GetPortfolioHistoryRequest(period="all", timeframe="1D")
        history = trading_client.get_portfolio_history(req)
        if history.equity:
            for e in history.equity:
                if e is not None and e > 0:
                    initial_capital = float(e)
                    break
    except Exception as e:
        print(f"Error fetching initial capital: {e}")

    return {
        "equity": float(account.equity),
        "buying_power": float(account.buying_power),
        "initial_capital": initial_capital,
        "positions": [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl)
            } for p in positions
        ]
    }

@app.get("/trades")
def get_trades(db: Session = Depends(get_db)):
    trades = db.query(models.Trade).order_by(models.Trade.timestamp.desc()).limit(50).all()
    
    # Sync with Alpaca
    if trading_client:
        try:
            # Fetch recent orders from Alpaca
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50)
            alpaca_orders = trading_client.get_orders(req)
            alpaca_map = {str(o.id): o for o in alpaca_orders}
            
            # Map existing local trades by order_id
            local_map = {t.order_id: t for t in trades if t.order_id}
            
            updates = False
            
            # 0. Cleanup Duplicates & Fix Data
            trades_to_delete = []
            for t1 in trades:
                if t1.reason == "External/Manual Order" and t1.order_id:
                    for t2 in trades:
                        if t1.id != t2.id and t2.reason != "External/Manual Order" and t1.symbol == t2.symbol and abs(t1.qty - t2.qty) < 0.001:
                            time_diff = abs((t1.timestamp - t2.timestamp).total_seconds())
                            if time_diff < 60:
                                print(f"Merging duplicate trade {t1.id} into {t2.id}")
                                t2.order_id = t1.order_id
                                t2.status = t1.status
                                t2.price = t1.price
                                trades_to_delete.append(t1)
                                updates = True
            
            for t in trades_to_delete:
                db.delete(t)
                if t in trades:
                    trades.remove(t)

            # Fix "ORDERSIDE.BUY" labels
            for t in trades:
                if t.side and "ORDERSIDE" in str(t.side).upper():
                    if "BUY" in str(t.side).upper():
                        t.side = "buy"
                    elif "SELL" in str(t.side).upper():
                        t.side = "sell"
                    updates = True

            # 1. Update existing trades
            for trade in trades:
                if trade.order_id and trade.order_id in alpaca_map:
                    ao = alpaca_map[trade.order_id]
                    new_status = ao.status.value if hasattr(ao.status, 'value') else str(ao.status)
                    if trade.status != new_status:
                        trade.status = new_status
                        updates = True
                    if ao.filled_qty is not None and float(ao.filled_qty) > 0:
                        trade.qty = float(ao.filled_qty)
                        if ao.filled_avg_price is not None:
                            trade.price = float(ao.filled_avg_price)
                        updates = True
            
            # 2. Import missing orders
            for order_id, ao in alpaca_map.items():
                if order_id not in local_map:
                    match_found = False
                    for t in trades:
                        if not t.order_id and t.symbol == ao.symbol and abs(t.qty - float(ao.qty or 0)) < 0.001:
                            print(f"Linking orphaned trade {t.id} to order {order_id}")
                            t.order_id = str(ao.id)
                            t.status = ao.status.value if hasattr(ao.status, 'value') else str(ao.status)
                            match_found = True
                            updates = True
                            break
                    
                    if not match_found:
                        exists = db.query(models.Trade).filter(models.Trade.order_id == order_id).first()
                        if not exists:
                            print(f"Importing missing order: {order_id}")
                            new_trade = models.Trade(
                                symbol=ao.symbol,
                                side=ao.side.value if hasattr(ao.side, 'value') else str(ao.side),
                                qty=float(ao.qty) if ao.qty else 0,
                                price=float(ao.filled_avg_price) if ao.filled_avg_price else 0,
                                timestamp=ao.created_at,
                                reason="External/Manual Order",
                                order_id=str(ao.id),
                                status=ao.status.value if hasattr(ao.status, 'value') else str(ao.status)
                            )
                            db.add(new_trade)
                            updates = True
            
            if updates:
                db.commit()
                trades = db.query(models.Trade).order_by(models.Trade.timestamp.desc()).limit(50).all()
                
        except Exception as e:
            print(f"Error syncing trades: {e}")
            
    return trades

@app.get("/logs")
async def get_logs(db: Session = Depends(get_db)):
    logs = db.query(models.AgentLog).order_by(models.AgentLog.timestamp.desc()).limit(50).all()
    # Broadcast logs to all WebSocket clients
    await broadcast_ws_message({"type": "logs", "data": [
        {"timestamp": str(log.timestamp), "title": log.title, "content": log.content} for log in logs
    ]})
    return logs

@app.get("/performance")
def get_performance(period: str = "1M", timeframe: str = None, db: Session = Depends(get_db)):
    if trading_client:
        try:
            if timeframe is None:
                if period == "1D":
                    timeframe = "5Min"
                elif period == "1W":
                    timeframe = "1H"
                else:
                    timeframe = "1D"

            req = GetPortfolioHistoryRequest(
                period=period,
                timeframe=timeframe
            )
            
            history = trading_client.get_portfolio_history(req)
            
            data = []
            for i in range(len(history.timestamp)):
                ts = history.timestamp[i]
                equity = history.equity[i]
                pnl = history.profit_loss[i] if history.profit_loss else 0
                
                if equity is None:
                    continue
                
                if period in ["1D", "1W"]:
                    date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                else:
                    date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                
                data.append({
                    "date": date_str,
                    "equity": equity,
                    "pnl": pnl
                })
            
            if period == "1D":
                try:
                    account = trading_client.get_account()
                    current_equity = float(account.equity)
                    last_equity = float(account.last_equity)
                    current_pnl = current_equity - last_equity
                    
                    now = datetime.datetime.now()
                    date_str = now.strftime('%Y-%m-%d %H:%M')
                    
                    if not data or data[-1]['date'] != date_str:
                        data.append({
                            "date": date_str,
                            "equity": current_equity,
                            "pnl": current_pnl
                        })
                except Exception as e:
                    print(f"Error fetching live account data: {e}")

            return data
        except Exception as e:
            print(f"Error fetching Alpaca history: {e}")
            pass

    performance = db.query(models.DailyEquity).order_by(models.DailyEquity.date.asc()).all()
    return performance


@app.get("/benchmark")
def get_benchmark(symbol: str = "QQQ", period: str = "1M", timeframe: str = None):
    if not data_client:
        raise HTTPException(status_code=503, detail="Alpaca data client not initialized")

    def _fetch_stooq_daily(us_symbol: str) -> list[dict]:
        """Open fallback: Stooq daily bars (no API key)."""
        stooq_symbol = f"{us_symbol.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")

        reader = csv.DictReader(io.StringIO(text))
        out: list[dict] = []
        for row in reader:
            d = row.get("Date")
            c = row.get("Close")
            if not d or not c:
                continue
            try:
                close = float(c)
            except ValueError:
                continue
            out.append({"date": d, "close": close})
        return out

    if timeframe is None:
        if period == "1D":
            tf = TimeFrame(5, TimeFrameUnit.Minute)
        elif period == "1W":
            tf = TimeFrame.Hour
        else:
            tf = TimeFrame.Day
    else:
        # Best-effort parsing for known values
        if timeframe in ["5Min", "5MIN", "5m"]:
            tf = TimeFrame(5, TimeFrameUnit.Minute)
        elif timeframe in ["1H", "1h", "1Hour"]:
            tf = TimeFrame.Hour
        else:
            tf = TimeFrame.Day

    now = datetime.datetime.now()
    if period == "1D":
        start_time = now - datetime.timedelta(days=1)
    elif period == "1W":
        start_time = now - datetime.timedelta(days=7)
    elif period == "1M":
        start_time = now - datetime.timedelta(days=30)
    elif period == "3M":
        start_time = now - datetime.timedelta(days=90)
    elif period == "1Y":
        start_time = now - datetime.timedelta(days=365)
    elif period == "ALL":
        # Keep it bounded for API limits; adjust if your plan supports more.
        start_time = now - datetime.timedelta(days=5 * 365)
    else:
        start_time = now - datetime.timedelta(days=30)

    try:
        df = None

        # Explicitly request a non-SIP feed to avoid subscription errors.
        # IEX is commonly available on free plans; DELAYED_SIP is a secondary fallback.
        for feed in (DataFeed.IEX, DataFeed.DELAYED_SIP):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=tf,
                    start=start_time,
                    end=now,
                    feed=feed,
                )
                bars = data_client.get_stock_bars(request)
                df = bars.df
                if df is not None and not df.empty:
                    break
            except Exception as inner_e:
                df = None

        if df is None or df.empty:
            # Open fallback (daily only): Stooq
            if tf == TimeFrame.Day:
                try:
                    stooq = _fetch_stooq_daily(symbol)
                    if not stooq:
                        return []

                    # Filter to the requested window
                    start_date = start_time.date()
                    end_date = now.date()
                    return [p for p in stooq if start_date <= datetime.date.fromisoformat(p["date"]) <= end_date]
                except Exception as stooq_e:
                    print(f"Error fetching benchmark data for {symbol} (stooq): {stooq_e}")
                    return []

            return []

        df = df.reset_index()
        data = []
        for _, row in df.iterrows():
            ts = row.get("timestamp")
            close = row.get("close")
            if ts is None or close is None:
                continue

            dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if period in ["1D", "1W"]:
                date_str = dt.strftime('%Y-%m-%d %H:%M')
            else:
                date_str = dt.strftime('%Y-%m-%d')

            data.append({
                "date": date_str,
                "close": float(close)
            })

        return data
    except Exception as e:
        print(f"Error fetching benchmark data for {symbol}: {e}")
        return []



@app.post("/positions/close")
async def close_positions(req: ClosePositionsRequest, db: Session = Depends(get_db)):
    if not trading_client:
        raise HTTPException(status_code=503, detail="Alpaca client not initialized")
    results = []
    for symbol in req.symbols:
        try:
            # Find position
            positions = trading_client.get_all_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)
            if not pos:
                results.append({"symbol": symbol, "status": "not found"})
                continue
            sell_req = MarketOrderRequest(
                symbol=symbol,
                qty=pos.qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            order = trading_client.submit_order(sell_req)
            results.append({"symbol": symbol, "status": "submitted", "order_id": str(order.id)})
        except Exception as e:
            results.append({"symbol": symbol, "status": "error", "error": str(e)})
    await trigger_state_broadcast()
    return {"results": results}


@app.post("/run-agent")
async def run_agent(db: Session = Depends(get_db)):
    try:
        await autonomous_cycle(db, force=True)
        await trigger_state_broadcast()
        return {"status": "Agent cycle started"}
    except asyncio.CancelledError:
        return JSONResponse({"detail": "Request cancelled"}, status_code=499)

def get_market_data(symbols):
    if not data_client:
        return {}
    
    if not symbols:
        return {}
    
    # Get last 5 days of data
    start_time = datetime.datetime.now() - datetime.timedelta(days=5)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_time
    )
    bars = data_client.get_stock_bars(request)
    return bars.df.to_json() if not bars.df.empty else "{}"

def get_super_advisor_insight():
    if not data_client or not grok_client:
        return "Market data or AI client unavailable. Proceed with caution."
    
    try:
        # Analyze broad market indices
        indices = ['SPY', 'QQQ', 'DIA', 'IWM']
        data = get_market_data(indices)
        
        prompt = f"""
        You are the 'Macro Sentinel'—a Senior Market Strategist for a high-frequency quant fund.
        
        Your input data is the recent price action (OHLCV) of major US Indices (SPY, QQQ, DIA, IWM):
        {data}
        
        Perform a technical audit on this data. Look for:
        1. Trend strength (e.g., higher highs, lower lows).
        2. Volatility compression or expansion.
        3. Relative strength (Is Tech/QQQ leading or lagging Small Caps/IWM?).
        
        Output a structured strategic directive in this format:
        
        MARKET REGIME: [Bullish / Bearish / Range-bound / Volatile]
        SENTIMENT SCORE: [0 (Max Fear) to 10 (Max Greed)]
        TACTICAL DIRECTIVE: [One concise paragraph. E.g., "The SPY is overextended. Prioritize taking profits on long positions. Look for rotation into IWM."]
        
        Be decisive. Your directive will control the risk parameters of the execution bot.
        """
        
        response = grok_client.chat.completions.create(
            model=LLM_SETTINGS["grok_model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=768
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Advisor error: {e}")
        return "Advisor unavailable. Trade based on local signals."

# --- Multi-Agent Helpers ---

async def call_grok(system_prompt, user_prompt):
    if agents.USE_LOCAL_FALLBACK or not grok_client:
        return await agents.call_local_llm(system_prompt, user_prompt)

    try:
        response = await grok_client.chat.completions.create(
            model=LLM_SETTINGS["grok_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=768
        )
        return json.loads(response.choices[0].message.content)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "402" in error_msg or "insufficient credits" in error_msg:
            print(f"Grok call failed (402): {e}. PERMANENTLY Switching to fallback model ({LOCAL_LLM_MODEL}).")
            agents.USE_LOCAL_FALLBACK = True
            return await agents.call_local_llm(system_prompt, user_prompt)
        else:
            print(f"Grok call failed: {e}")
            return {}

async def manage_existing_positions(db: Session):
    """
    Audits all open positions and closes them if they hit Take Profit (5%) or Stop Loss (-3%).
    """
    if not trading_client: return

    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            symbol = pos.symbol
            # Use total unrealized P/L % instead of intraday
            unrealized_plpc = float(pos.unrealized_plpc) if hasattr(pos, 'unrealized_plpc') else 0.0
            
            # TP/SL Thresholds
            tp_threshold = TAKE_PROFIT_PERCENTAGE
            sl_threshold = STOP_LOSS_PERCENTAGE
            
            if unrealized_plpc >= tp_threshold or unrealized_plpc <= sl_threshold:
                print(f"[AUDIT] {symbol} hit threshold ({round(unrealized_plpc*100, 2)}%). Calling PositionMonitor...")
                
                pos_data = {
                    "symbol": symbol,
                    "qty": pos.qty,
                    "current_price": pos.current_price,
                    "avg_entry": pos.avg_entry_price,
                    "unrealized_plpc": round(unrealized_plpc*100, 2),
                    "tp_threshold": tp_threshold*100,
                    "sl_threshold": sl_threshold*100
                }
                market_context = "High frequency audit. Decide FULL_CLOSE or PARTIAL_CLOSE based on profit."
                
                monitor = agents.PositionMonitor()
                try:
                    decision = await monitor.monitor_position(pos_data, market_context)
                    action = decision.get("action", "HOLD")
                    close_fraction = float(decision.get("close_fraction", 1.0))
                    reasoning = decision.get("reasoning", "No reasoning provided.")
                    
                    if action in ["FULL_CLOSE", "PARTIAL_CLOSE"]:
                        sell_qty = int(float(pos.qty) * (close_fraction if action == "PARTIAL_CLOSE" else 1.0))
                        if sell_qty < 1:
                            sell_qty = 1 
                            
                        sell_req = MarketOrderRequest(
                            symbol=symbol,
                            qty=str(sell_qty),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        
                        trading_client.submit_order(sell_req)
                        
                        log = models.AgentLog(
                            title=f"{action}: {symbol}",
                            content=f"Closed {sell_qty} shares of {symbol} at {round(unrealized_plpc*100, 2)}% P/L.\nReason: {reasoning}"
                        )
                        db.add(log)
                        db.commit()
                        
                        await broadcast_ws_message({
                            "type": "logs",
                            "data": [{"timestamp": str(datetime.datetime.now()), "title": log.title, "content": log.content}]
                        })
                except Exception as e:
                    print(f"Failed to execute PositionMonitor for {symbol}: {e}")
                    
    except Exception as e:
        print(f"Error in position management: {e}")

async def autonomous_cycle(db: Session = None, force: bool = False):
    if not BOT_ACTIVE and not force:
        print("Bot is paused. Skipping cycle.")
        return

    if not db:
        db = SessionLocal()
    
    print("Running MULTI-AGENT autonomous cycle...")
    if not trading_client:
        print("Trading client not initialized")
        return

    if not grok_client and not agents.USE_LOCAL_FALLBACK:
        print("LLM client not initialized")
        return

    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        last_equity = float(account.last_equity)
        daily_drawdown = (equity - last_equity) / last_equity if last_equity > 0 else 0
        
        # 0. GLOBAL CIRCUIT BREAKER (Kill Switch)
        if daily_drawdown < -0.03: # 3% Daily Drawdown Limit
            print(f"!!! CIRCUIT BREAKER TRIGGERED: Daily Drawdown is {round(daily_drawdown*100, 2)}% !!!")
            # Log the event
            log = models.AgentLog(
                title="CIRCUIT BREAKER: KILL SWITCH",
                content=f"Daily drawdown reached {round(daily_drawdown*100, 2)}%. Closing all positions to protect capital."
            )
            db.add(log)
            db.commit()
            
            # Close all positions
            trading_client.close_all_positions(cancel_orders=True)
            return

        # --- POSITION AUDIT PHASE (Take Profit / Stop Loss) ---
        await manage_existing_positions(db)

        # 1. GATHER DATA FOR REGIME ARBITER
        vix = 20.0 
        try:
            req = StockSnapshotRequest(symbol_or_symbols=["VIX"])
            # In Alpaca, VIX might not be available in snapshots directly for all. 
            # Use a conservative 20 if not found.
        except: pass

        # Get macro news summary
        active_symbols = get_active_stocks(limit=5)
        news_summary = ""
        for s in active_symbols:
            news = get_latest_news(s, max_results=2)
            for n in news:
                news_summary += f"{s}: {n['title']} | "

        macro_web_summary = ""
        if WEB_RESEARCH_ENABLED:
            macro_web = get_macro_web_research(
                max_results=WEB_RESEARCH_MACRO_MAX_RESULTS,
                days=min(WEB_RESEARCH_DAYS, 3),
            )
            macro_web_summary = format_tavily_for_prompt("Macro", macro_web, max_items=5)

        combined_regime_context = f"{news_summary}\n{macro_web_summary}".strip()

        # 1. REGIME ARBITER PHASE
        arbiter = agents.RegimeArbiter(grok_client, model=LLM_SETTINGS["grok_model"])
        indices_data = get_market_data(['SPY', 'QQQ', 'IWM'])
        # Truncate market_snapshot data to prevent context overflow with small local models
        if len(indices_data) > 2000:
            indices_data = indices_data[:2000] + "... (truncated)"
        
        arbiter_response = await arbiter.determine_regime(
            market_snapshot=indices_data,
            sentiment_summary=combined_regime_context[:700],
            vix=vix
        )
        current_regime = arbiter_response.get("regime", "TRENDING")
        primary_strategy = arbiter_response.get("primary_strategy", "Momentum")
        
        print(f"ARBITER DECISION: Regime={current_regime}, Strategy={primary_strategy}")
        
        # Log Arbiter thought
        arb_log = models.AgentLog(
            title=f"REGIME ARBITER: {current_regime}",
            content=f"Strategy: {primary_strategy}\nReasoning: {arbiter_response.get('reasoning')}"
        )
        db.add(arb_log)
        db.commit()

        if current_regime == "VOLATILE_UNRELIABLE" or primary_strategy == "Cash Only":
            print("Arbiter suggests caution, but proceeding with opportunistic trades.")
            # Force a strategy if none valid provided
            if primary_strategy == "Cash Only":
                primary_strategy = "Mean Reversion"

        # 2. ANALYST & ADVERSARIAL PHASE
        current_positions = trading_client.get_all_positions()
        
        if len(current_positions) >= 5:
            print(f"Max concurrent trades (5) reached. Skipping new trade search.")
            return
            
        candidates = get_active_stocks(limit=10)
        if not candidates:
            candidates = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMD', 'META']

        web_research_symbols = set()
        if WEB_RESEARCH_ENABLED:
            web_research_symbols = set(
                shortlist_candidates_for_web_research(candidates, max_count=WEB_RESEARCH_MAX_TICKERS)
            )

        for ticker in candidates:

            try:
                # Perception upgrades already integrated (Sentiment + OFI)
                snap_req = StockSnapshotRequest(symbol_or_symbols=[ticker])
                snap = data_client.get_stock_snapshot(snap_req)[ticker]
                price = snap.latest_trade.price
                rsi = 50.0

                # --- News and Sentiment Integration ---
                ticker_news = get_latest_news(ticker, max_results=3)
                news_prompt = format_news_for_prompt(ticker, ticker_news)
                social_sentiment = get_social_sentiment(ticker, max_results=3)
                sentiment_prompt = format_sentiment_for_prompt(ticker, social_sentiment)
                tavily_prompt = ""
                if ticker in web_research_symbols:
                    web_hits = get_ticker_web_research(
                        ticker,
                        max_results=WEB_RESEARCH_TICKER_MAX_RESULTS,
                        days=WEB_RESEARCH_DAYS,
                    )
                    tavily_prompt = format_tavily_for_prompt(ticker, web_hits, max_items=5)

                # Combine news and sentiment for analysis
                combined_perception = f"{news_prompt}\n{sentiment_prompt}\n{tavily_prompt}".strip()

                # Optionally, pass this to the Analyst agent or use in trading logic
                sentiment_agent = agents.SentimentAgent(grok_client, model=LLM_SETTINGS["grok_model"])
                sentiment_result = await sentiment_agent.analyze_sentiment(ticker, combined_perception)
                sentiment_analysis = f"Score: {sentiment_result.get('sentiment_score', 0)} | Narrative: {sentiment_result.get('narrative', 'N/A')}"

                order_flow_desc = "Neutral"
                if hasattr(snap, 'latest_quote') and snap.latest_quote:
                    bid_sz = snap.latest_quote.bid_size
                    ask_sz = snap.latest_quote.ask_size
                    if (bid_sz + ask_sz) > 0:
                        imbalance = (bid_sz - ask_sz) / (bid_sz + ask_sz)
                        if imbalance > 0.2:
                            order_flow_desc = f"Strong Bullish Imbalance ({round(imbalance, 2)})"
                        elif imbalance < -0.2:
                            order_flow_desc = f"Strong Bearish Imbalance ({round(imbalance, 2)})"
                        else:
                            order_flow_desc = f"Neutral Imbalance ({round(imbalance, 2)})"

                price_data = {
                    'price': price,
                    'rsi': rsi,
                    'order_flow': order_flow_desc,
                    'market_context': f"{current_regime} - {primary_strategy}",
                    'sentiment_analysis': sentiment_analysis,
                    'news': news_prompt,
                    'social_sentiment': sentiment_prompt,
                    'web_research': tavily_prompt,
                }

                analyst_agent = agents.Analyst(grok_client, model=LLM_SETTINGS["grok_model"])
                analyst_response = await analyst_agent.analyze_ticker(ticker, price_data)

                if not analyst_response or analyst_response.get("signal") not in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"]:
                    continue

                signal = analyst_response["signal"]
                conviction = analyst_response.get("conviction_score", 0.5)
                thesis = analyst_response.get("technical_thesis", "")
                if tavily_prompt:
                    thesis = f"{thesis}\n\nExternal web evidence:\n{tavily_prompt[:700]}"

                # 3. ADVERSARIAL CHALLENGE
                adversary = agents.AdversarialAgent(grok_client, model=LLM_SETTINGS["grok_model"])
                adversary_response = await adversary.challenge_trade(ticker, signal, thesis, price)
                bear_case = adversary_response.get("bear_case", "No major counter-risks identified.")
                
                risk_level = adversary_response.get('counter_risk_level')
                print(f"ADVERSARY for {ticker}: Risk={risk_level}")

                # Hard Veto only on TERMINAL, or HIGH if flag is explicitly set
                if adversary_response.get("invalid_thesis_flag"):
                    if risk_level == "TERMINAL":
                        print(f"Trade for {ticker} VETOED by Adversary (TERMINAL Risk).")
                        continue
                    elif risk_level == "HIGH":
                        print(f"Trade for {ticker} VETOED by Adversary (HIGH Risk).")
                        continue
                    else:
                        print(f"Adversary flag ignored for {ticker} (Risk is {risk_level}). Proceeding...")

                # 4. RISK MANAGER PHASE
                target_value = equity * 0.05 
                requested_qty = int(target_value / price) if price > 0 else 0
                
                if requested_qty == 0 and "BUY" in signal: continue

                risk_prompt = agent_prompts.RISK_MANAGER_TASK_TEMPLATE.format(
                    signal=signal,
                    conviction=conviction,
                    ticker=ticker,
                    requested_qty=requested_qty,
                    entry_price=price,
                    stop_price=analyst_response.get("suggested_stop_loss", price * 0.95),
                    current_regime=current_regime,
                    equity=equity,
                    buying_power=buying_power
                )
                
                # Update Risk Manager with Bear Case insight
                risk_sys_prompt = agent_prompts.RISK_MANAGER_SYSTEM_PROMPT.format(max_pos_size_pct=10, max_total_exposure=1.5)
                risk_sys_prompt += f"\n\nCONSIDER BEAR CASE: {bear_case}"

                risk_agent = agents.RiskManager(grok_client, model=LLM_SETTINGS["grok_model"])
                risk_response = await call_grok(risk_sys_prompt, risk_prompt) # Using call_grok directly due to custom sys_prompt injection

                
                if risk_response and risk_response.get("decision") in ["APPROVED", "MODIFIED"]:
                    final_qty = risk_response.get("final_qty", requested_qty)
                    if final_qty <= 0: continue
                    
                    side = 'buy' if "BUY" in signal else 'sell'
                    
                    # --- Short Selling Guardrail ---
                    if side == 'sell':
                        try:
                            # 1. Check if we already have a long position to liquidate
                            has_pos = False
                            try:
                                trading_client.get_open_position(ticker)
                                has_pos = True
                            except:
                                has_pos = False # No position
                            
                            # 2. If no position, this is a short sell attempt. Check shortability.
                            if not has_pos:
                                asset = trading_client.get_asset(ticker)
                                if not asset.shortable:
                                    print(f"GUARDRAIL: Skipping SHORT on {ticker} (Asset not shortable at Alpaca)")
                                    log = models.AgentLog(
                                        title=f"GUARDRAIL: {ticker}",
                                        content=f"Skipped short sell for {ticker} - Asset is NOT shortable."
                                    )
                                    db.add(log)
                                    db.commit()
                                    continue
                                if not asset.easy_to_borrow:
                                    print(f"GUARDRAIL: {ticker} is Hard-to-Borrow. Order may still fail later.")
                        except Exception as e:
                            print(f"Guardrail error for {ticker}: {e}")
                            # Continue anyway, let Alpaca handle the final rejection if needed
                    
                    reason = f"{ticker} {signal} | Conv: {conviction} | Risk: {risk_response.get('risk_analysis')}"
                    
                    print(f"RISK APPROVED: {side.upper()} {final_qty} {ticker}")
                    
                    # Execution with Midpoint Limit logic to minimize slippage
                    limit_price = price # Default to last price
                    if hasattr(snap, 'latest_quote') and snap.latest_quote:
                        bid = snap.latest_quote.bid_price
                        ask = snap.latest_quote.ask_price
                        if bid > 0 and ask > 0:
                            limit_price = round((bid + ask) / 2, 2)
                    
                    from alpaca.trading.requests import LimitOrderRequest
                    order_data = LimitOrderRequest(
                        symbol=ticker,
                        qty=final_qty,
                        side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                        limit_price=limit_price,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    try:
                        order = trading_client.submit_order(order_data=order_data)
                        print(f"Limit Order submitted: {order.id} at {limit_price}")
                        
                        # Log Trade
                        trade = models.Trade(
                            symbol=ticker,
                            side=side,
                            qty=final_qty,
                            price=price,
                            reason=reason,
                            order_id=str(order.id),
                            status=order.status.value if hasattr(order.status, 'value') else str(order.status)
                        )
                        db.add(trade)
                        db.commit()
                        
                        # Broadcast
                        await broadcast_ws_message({
                            "type": "trades",
                            "data": [{
                                "timestamp": str(datetime.datetime.now()),
                                "side": side,
                                "symbol": ticker,
                                "qty": final_qty,
                                "price": price,
                                "reason": reason,
                                "status": trade.status
                            }]
                        })
                    except Exception as e:
                        print(f"Order failed for {ticker}: {e}")

                # Log thoughts for analyst/risk for each candidate if significant
                if analyst_response:
                    log = models.AgentLog(
                        title=f"ANALYSIS: {ticker}",
                        content=f"Signal: {signal}\nConviction: {conviction}\nRisk Decision: {risk_response.get('decision') if risk_response else 'N/A'}\n{analyst_response.get('technical_thesis', '')}"
                    )
                    db.add(log)
                    db.commit()
                    await broadcast_ws_message({
                        "type": "logs",
                        "data": [{"timestamp": str(datetime.datetime.now()), "title": log.title, "content": log.content}]
                    })

            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        # Log Daily Equity
        daily_equity = models.DailyEquity(
            equity=equity,
            pnl=equity - float(account.last_equity) 
        )
        db.add(daily_equity)
        db.commit()

    except Exception as e:
        print(f"Autonomous cycle error: {e}")
    finally:
        if db:
            db.close()

# Schedule the autonomous cycle
# scheduler.add_job(autonomous_cycle, 'interval', minutes=30)
scheduler.add_job(autonomous_cycle, 'cron', day_of_week='mon-fri', hour=9, minute=30, timezone='America/New_York')