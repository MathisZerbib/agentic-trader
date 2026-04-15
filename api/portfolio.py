import datetime
import csv
import urllib.request
import io

import json
import os
from typing import List, Optional
from core.clients import trading_client, data_client
from alpaca.trading.requests import GetPortfolioHistoryRequest, GetOrdersRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.enums import QueryOrderStatus

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from api.ws import broadcast_ws_message
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models
from schemas import ClosePositionsRequest
from services.state import trigger_state_broadcast

router = APIRouter()

@router.get("/portfolio")
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

@router.get("/trades")
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

@router.get("/logs")
async def get_logs(db: Session = Depends(get_db)):
    logs = db.query(models.AgentLog).order_by(models.AgentLog.timestamp.desc()).limit(50).all()
    # Broadcast logs to all WebSocket clients
    await broadcast_ws_message({"type": "logs", "data": [
        {"timestamp": str(log.timestamp), "title": log.title, "content": log.content} for log in logs
    ]})
    return logs

@router.get("/performance")
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


@router.get("/benchmark")
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
                print(f"DataFeed {feed} error: {inner_e}")
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



@router.post("/positions/close")
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


