import datetime
from alpaca.data.requests import StockSnapshotRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import numpy as np

def get_ticker_metrics(ticker, data_client=None):
    """
    Fetches price, RSI, EMA distance, and order flow imbalance for a ticker.
    This is a stub; you should expand with real calculations as needed.
    """
    if not data_client:
        return {
            'price': 0.0,
            'rsi': 50,
            'ema_diff': 0,
            'order_flow': 0
        }
    try:
        # Get latest price from snapshot
        req = StockSnapshotRequest(symbol_or_symbols=[ticker])
        snapshot = data_client.get_stock_snapshot(req)
        price = snapshot[ticker].latest_trade.price if ticker in snapshot and snapshot[ticker].latest_trade else 0.0

        # Get last 200 bars (for EMA200)
        bars_req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            limit=200
        )
        bars = data_client.get_stock_bars(bars_req)
        df = bars.df
        if df.empty or ticker not in df.index.get_level_values(0):
            return {
                'price': price,
                'rsi': 50,
                'ema_diff': 0,
                'order_flow': 0
            }
        tdf = df.xs(ticker)
        closes = tdf['close'].values
        # --- RSI Calculation ---
        def calc_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50
            deltas = np.diff(prices)
            seed = deltas[:period]
            up = seed[seed > 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 0
            rsi = 100. - 100. / (1. + rs)
            for i in range(period, len(prices) - 1):
                delta = deltas[i]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta
                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                rs = up / down if down != 0 else 0
                rsi = 100. - 100. / (1. + rs)
            return float(np.round(rsi, 2))
        rsi = calc_rsi(closes)

        # --- EMA 200 Calculation ---
        def calc_ema(prices, period=200):
            if len(prices) < period:
                return prices[-1]
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            a = np.convolve(prices, weights, mode='valid')
            return a[-1]
        ema_200 = calc_ema(closes, 200)
        ema_diff = ((price - ema_200) / ema_200) * 100 if ema_200 else 0

        # --- Order Flow Imbalance (simple: up volume - down volume) ---
        up_vol = tdf[tdf['close'] > tdf['open']]['volume'].sum()
        down_vol = tdf[tdf['close'] < tdf['open']]['volume'].sum()
        order_flow = float(up_vol - down_vol)

        return {
            'price': price,
            'rsi': rsi,
            'ema_diff': float(np.round(ema_diff, 2)),
            'order_flow': order_flow
        }
    except Exception as e:
        print(f"Error fetching metrics for {ticker}: {e}")
        return {
            'price': 0.0,
            'rsi': 50,
            'ema_diff': 0,
            'order_flow': 0
        }
