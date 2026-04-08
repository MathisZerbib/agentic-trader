import re

with open("backend/agents/orchestrator.py", "r") as f:
    content = f.read()

# Fix the NameErrors for TAKE_PROFIT_PERCENTAGE
content = content.replace("tp_threshold = TAKE_PROFIT_PERCENTAGE", "tp_threshold = settings.TAKE_PROFIT_PERCENTAGE")
content = content.replace("sl_threshold = STOP_LOSS_PERCENTAGE", "sl_threshold = settings.STOP_LOSS_PERCENTAGE")

# Add the limitation counter to the candidate loop
target_loop_start = """        if len(current_positions) >= 5:
            print(f"Max concurrent trades (5) reached. Skipping new trade search.")
            return
            
        candidates = get_active_stocks(limit=10)"""

new_loop_start = """        open_positions_count = len(current_positions)
        if open_positions_count >= 5:
            print(f"Max concurrent trades (5) reached. Skipping new trade search.")
            return
            
        remaining_slots = 5 - open_positions_count
        
        candidates = get_active_stocks(limit=10)"""

content = content.replace(target_loop_start, new_loop_start)

# In the actual order submission:
order_submit = """                    try:
                        order = trading_client.submit_order(order_data=order_data)
                        print(f"Limit Order submitted: {order.id} at {limit_price}")
                        
                        # Log Trade
                        trade = models.Trade("""

new_order_submit = """                    try:
                        order = trading_client.submit_order(order_data=order_data)
                        print(f"Limit Order submitted: {order.id} at {limit_price}")
                        
                        remaining_slots -= 1
                        
                        # Log Trade
                        trade = models.Trade("""
content = content.replace(order_submit, new_order_submit)

# At the end of the candidate loop
order_end = """                        await broadcast_ws_message({
                            "type": "trades",
                            "data": [{"timestamp": str(datetime.datetime.now()), "symbol": trade.symbol, "side": trade.side, "qty": trade.qty, "price": trade.price}]
                        })
                    except Exception as e:
                        print(f"Failed to submit order for {ticker}: {e}")"""

new_order_end = """                        await broadcast_ws_message({
                            "type": "trades",
                            "data": [{"timestamp": str(datetime.datetime.now()), "symbol": trade.symbol, "side": trade.side, "qty": trade.qty, "price": trade.price}]
                        })
                        
                        if remaining_slots <= 0:
                            print("Reached max 5 open positions during cycle. Halting further trades.")
                            break
                    except Exception as e:
                        print(f"Failed to submit order for {ticker}: {e}")"""
content = content.replace(order_end, new_order_end)

with open("backend/agents/orchestrator.py", "w") as f:
    f.write(content)
