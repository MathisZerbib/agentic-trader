from sqlalchemy.orm import Session
from alpaca.data.requests import StockSnapshotRequest
import models
from database import SessionLocal
from core.clients import trading_client, data_client, bot_state

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
                        "unrealized_plpc": (float(p.unrealized_plpc) * 100) if p.unrealized_plpc is not None else 
                                           ((float(p.unrealized_pl) / (float(p.avg_entry_price) * float(p.qty))) * 100 if float(p.avg_entry_price) > 0 else 0.0),
                        "current_price": float(p.current_price) if hasattr(p, 'current_price') else 0.0,
                        "change_today": float(p.unrealized_intraday_plpc or 0.0) * 100
                    } for p in positions
                ]
            }
        except Exception as e:
            print(f"Error fetching portfolio state: {e}")
            portfolio_data = None

    return {
        "type": "state",
        "bot_active": bot_state["BOT_ACTIVE"],
        "trading_locked": bot_state.get("TRADING_LOCKED", False),
        "market_status": market_status,
        "next_open": next_open,
        "next_close": next_close,
        "qqq_change": round(qqq_change, 2),
        "portfolio": portfolio_data
    }

async def trigger_state_broadcast():
    from api.ws import broadcast_ws_message
    db = SessionLocal()
    try:
        state = get_current_state(db)
        await broadcast_ws_message(state)
    finally:
        db.close()

# Background broadcast for the scheduler
async def scheduled_broadcast():
    await trigger_state_broadcast()
