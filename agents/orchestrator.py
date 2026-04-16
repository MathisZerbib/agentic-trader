import asyncio
import traceback
import json
from datetime import datetime
from core.config import settings
from sqlalchemy.orm import Session
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from core.clients import trading_client, data_client, grok_client, bot_state
from api.ws import broadcast_ws_message
from database import SessionLocal
import models
import agents.agents as agents
import agents.prompts as agent_prompts
from services.market_data import (
    get_active_stocks, get_latest_news, get_market_data,
    get_social_sentiment, format_news_for_prompt,
    format_sentiment_for_prompt, get_ticker_web_research,
    get_macro_web_research, format_web_research_for_prompt,
    shortlist_candidates_for_web_research, get_rsi
)
from alpaca.data.requests import StockSnapshotRequest

trading_lock = asyncio.Lock()

async def call_grok(system_prompt, user_prompt):
    if agents.USE_LOCAL_FALLBACK or not grok_client:
        return await agents.call_local_llm(system_prompt, user_prompt)

    try:
        response = await grok_client.chat.completions.create(
            model=settings.DEFAULT_GROK_MODEL,
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
            print(f"Grok call failed (402): {e}. PERMANENTLY Switching to fallback model ({agents.LOCAL_LLM_MODEL}).")
            agents.USE_LOCAL_FALLBACK = True
            return await agents.call_local_llm(system_prompt, user_prompt)
        else:
            print(f"Grok call failed: {e}")
            return {}

async def manage_existing_positions(db: Session = None):
    """
    Audits all open positions and closes them if they hit Take Profit (5%) or Stop Loss (-3%).
    """
    if not trading_client: return

    if not db:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False

    try:
        # 1. Cleanup stale pending orders (> 3 mins)
        open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for order in open_orders:
            # Check age in seconds
            age_seconds = (datetime.now(order.created_at.tzinfo) - order.created_at).total_seconds()
            if age_seconds > 180: # 3 minutes
                print(f"[AUDIT] Canceling stale order {order.id} for {order.symbol} (Age: {int(age_seconds)}s)")
                trading_client.cancel_order_by_id(order.id)
                
                log = models.AgentLog(
                    title=f"STALE ORDER CANCELED: {order.symbol}",
                    content=f"Canceled pending order for {order.symbol} because it was older than 3 minutes ({int(age_seconds)}s)."
                )
                db.add(log)
                db.commit()

        # 2. Audit existing positions
        positions = trading_client.get_all_positions()
        for pos in positions:
            symbol = pos.symbol
            # Use total unrealized P/L % instead of intraday
            unrealized_plpc = float(pos.unrealized_plpc) if hasattr(pos, 'unrealized_plpc') else 0.0
            
            # TP/SL Thresholds
            tp_threshold = settings.TAKE_PROFIT_PERCENTAGE
            sl_threshold = settings.STOP_LOSS_PERCENTAGE
            
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
                
                monitor = agents.PositionMonitor(grok_client, model=settings.DEFAULT_GROK_MODEL)
                try:
                    decision = await monitor.monitor_position(pos_data, market_context)
                    if not decision: decision = {}
                    action = decision.get("action", "HOLD")
                    close_fraction = float(decision.get("close_fraction") or 1.0)
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
                        
                        try:
                            await broadcast_ws_message({
                                "type": "logs",
                                "data": [{"timestamp": str(datetime.now()), "title": log.title, "content": log.content}]
                            })
                        except:
                            pass
                except Exception as e:
                    print(f"Failed to execute PositionMonitor for {symbol}: {e}")
                    
    except Exception as e:
        print(f"Error in position management: {e}")
    finally:
        if should_close and db:
            db.close()
    
    # Check if we should refill positions (Opportunistic Homing)
    if bot_state.get('BOT_ACTIVE', False):
        try:
            current_positions = trading_client.get_all_positions()
            if len(current_positions) < 5:
                print(f"[REFILL] Slots available ({len(current_positions)}/5). Triggering opportunistic trade hunt...")
                # We use a task to avoid blocking the audit heartbeat
                asyncio.create_task(autonomous_cycle(skip_audit=True))
        except Exception as e:
            print(f"Refill check failed: {e}")

async def autonomous_cycle(db: Session = None, force: bool = False, skip_audit: bool = False):
    print("DEBUG: starting autonomous cycle V2")
    if not bot_state.get('BOT_ACTIVE', False) and not force:
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
        account = await asyncio.to_thread(trading_client.get_account)
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        last_equity = float(account.last_equity)
        daily_drawdown = (equity - last_equity) / last_equity if last_equity > 0 else 0
        
        # 0. GLOBAL CIRCUIT BREAKER (Kill Switch)
        if daily_drawdown < settings.DAILY_DRAWDOWN_THRESHOLD:
            # BYPASS: Allow if human-triggered AND no open positions (manual recovery)
            open_positions = await asyncio.to_thread(trading_client.get_all_positions)
            if force and len(open_positions) <= 3:
                print(f"CIRCUIT BREAKER: Manual bypass active (Drawdown: {round(daily_drawdown*100, 2)}%)")
            else:
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
        if not skip_audit:
            await manage_existing_positions(db)

        # 1. GATHER DATA FOR REGIME ARBITER
        vix = 20.0 
        try:
            req = StockSnapshotRequest(symbol_or_symbols=["VIX"])
            # In Alpaca, VIX might not be available in snapshots directly for all. 
            # Use a conservative 20 if not found.
        except: pass

        # Get macro news summary
        active_symbols = await asyncio.to_thread(get_active_stocks, limit=5)
        news_summary = ""
        for s in active_symbols:
            news = await asyncio.to_thread(get_latest_news, s, max_results=2)
            for n in news:
                news_summary += f"{s}: {n['title']} | "

        macro_web_summary = ""
        if settings.WEB_RESEARCH_ENABLED:
            macro_web = await asyncio.to_thread(
                get_macro_web_research,
                max_results=settings.WEB_RESEARCH_MACRO_MAX_RESULTS,
                days=min(settings.WEB_RESEARCH_DAYS, 3),
            )
            macro_web_summary = format_web_research_for_prompt("Macro", macro_web, max_items=5)

        combined_regime_context = f"{news_summary}\n{macro_web_summary}".strip()

        # 1. REGIME ARBITER PHASE
        arbiter = agents.RegimeArbiter(grok_client, model=settings.DEFAULT_GROK_MODEL)
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
        async with trading_lock:
            if bot_state.get('TRADING_LOCKED', False):
                print("TRADING IS LOCKED. Bypassing new trade search.")
                return

            current_positions = await asyncio.to_thread(trading_client.get_all_positions)
            # Factor in pending BUY orders as already 'occupying' a slot to prevent over-trading
            open_orders = await asyncio.to_thread(trading_client.get_orders, GetOrdersRequest(status=QueryOrderStatus.OPEN))
            pending_buys = [o for o in open_orders if o.side == OrderSide.BUY]
            
            total_slots_used = len(current_positions) + len(pending_buys)
            
            if total_slots_used >= 5:
                print(f"Max concurrent trades (5) reached. (Positions={len(current_positions)}, Pending Buys={len(pending_buys)}). Skipping new trade search.")
                return
                
            remaining_slots = 5 - total_slots_used
        
        candidates = await asyncio.to_thread(get_active_stocks, limit=10)
        if not candidates:
            candidates = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMD', 'META']

        web_research_symbols = set()
        if settings.WEB_RESEARCH_ENABLED:
            web_research_symbols = set(
                shortlist_candidates_for_web_research(candidates, max_count=settings.WEB_RESEARCH_MAX_TICKERS)
            )

        for ticker in candidates:

            try:
                # Perception upgrades already integrated (Sentiment + OFI)
                snap_req = StockSnapshotRequest(symbol_or_symbols=[ticker])
                snap = data_client.get_stock_snapshot(snap_req)[ticker]
                price = snap.latest_trade.price
                rsi = await asyncio.to_thread(get_rsi, ticker)

                # --- News and Sentiment Integration ---
                ticker_news = await asyncio.to_thread(get_latest_news, ticker, max_results=3)
                news_prompt = format_news_for_prompt(ticker, ticker_news)
                social_sentiment = await asyncio.to_thread(get_social_sentiment, ticker, max_results=3)
                sentiment_prompt = format_sentiment_for_prompt(ticker, social_sentiment)
                web_research_prompt = ""
                if ticker in web_research_symbols:
                    web_hits = await asyncio.to_thread(
                        get_ticker_web_research,
                        ticker,
                        max_results=settings.WEB_RESEARCH_TICKER_MAX_RESULTS,
                        days=settings.WEB_RESEARCH_DAYS,
                    )
                    web_research_prompt = format_web_research_for_prompt(ticker, web_hits, max_items=5)

                # Combine news and sentiment for analysis
                combined_perception = f"{news_prompt}\n{sentiment_prompt}\n{web_research_prompt}".strip()

                # Optionally, pass this to the Analyst agent or use in trading logic
                sentiment_agent = agents.SentimentAgent(grok_client, model=settings.DEFAULT_GROK_MODEL)
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
                    'web_research': web_research_prompt,
                }

                analyst_agent = agents.Analyst(grok_client, model=settings.DEFAULT_GROK_MODEL)
                analyst_response = await analyst_agent.analyze_ticker(ticker, price_data)

                if not analyst_response or analyst_response.get("signal") not in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"]:
                    if analyst_response and analyst_response.get("signal") == "WAIT":
                        print(f"Analyst says WAIT for {ticker}. Reasoning: {analyst_response.get('technical_thesis', 'No thesis provided.')}")
                        # Optional: Log the WAIT decision to DB so user sees it
                        wait_log = models.AgentLog(
                            title=f"ANALYSIS: {ticker} - WAIT",
                            content=f"Technical thesis suggests waiting.\nReasoning: {analyst_response.get('technical_thesis')}"
                        )
                        db.add(wait_log)
                        db.commit()
                    continue

                signal = analyst_response["signal"]
                conviction = analyst_response.get("conviction_score", 0.5)
                thesis = analyst_response.get("technical_thesis", "")
                if web_research_prompt:
                    thesis = f"{thesis}\n\nExternal web evidence:\n{web_research_prompt[:700]}"

                # 3. ADVERSARIAL CHALLENGE
                adversary = agents.AdversarialAgent(grok_client, model=settings.DEFAULT_GROK_MODEL)
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

                risk_agent = agents.RiskManager(grok_client, model=settings.DEFAULT_GROK_MODEL)
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
                        
                        remaining_slots -= 1
                        
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
                                "timestamp": str(datetime.now()),
                                "side": side,
                                "symbol": ticker,
                                "qty": final_qty,
                                "price": price,
                                "reason": reason,
                                "status": trade.status
                            }]
                        })
                        
                        if remaining_slots <= 0:
                            print("Reached max 5 open positions during cycle. Halting further trades.")
                            break
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
                    try:
                        await broadcast_ws_message({
                            "type": "logs",
                            "data": [{"timestamp": str(datetime.now()), "title": log.title, "content": log.content}]
                        })
                    except:
                        pass

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
        traceback.print_exc()
    finally:
        if db:
            db.close()

# Schedule the autonomous cycle
