import os
import time
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import NewsRequest, MostActivesRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame as BarTimeframe
from alpaca.data.enums import MostActivesBy, DataFeed
from datetime import datetime, timedelta
from services.llm_utils import get_active_local_model_sync


load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY")

_LMSTUDIO_PLAYWRIGHT_SEARCH_URL = os.getenv("LOCAL_LLM_URL", "http://host.docker.internal:1234")
_WEB_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_WEB_TTL_SECONDS = int(os.getenv("WEB_RESEARCH_TTL_SEC", "900"))
_WEB_TIMEOUT_SECONDS = float(os.getenv("WEB_RESEARCH_TIMEOUT_SEC", "120")) # Give LM Studio time to browse

news_client = None
screener_client = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    news_client = NewsClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
    screener_client = ScreenerClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)

def get_active_stocks(limit=5):
    """
    Fetches the most active stocks by volume.
    """
    if not screener_client:
        print("Alpaca ScreenerClient not initialized.")
        return []
    
    try:
        request_params = MostActivesRequest(by=MostActivesBy.VOLUME, top=limit)
        actives = screener_client.get_most_actives(request_params)
        return [item.symbol for item in actives.most_actives]
    except Exception as e:
        print(f"Error fetching active stocks: {e}")
        return []

def get_latest_news(symbol: str, max_results=3):
    """
    Fetches the latest news for a given stock symbol using Alpaca News API.
    """
    if not news_client:
        print("Alpaca NewsClient not initialized.")
        return []

    try:
        request_params = NewsRequest(
            symbols=symbol,
            limit=max_results
        )
        news_list = news_client.get_news(request_params)
        
        results = []
        # Access the list of news articles from the 'news' key in the data dictionary
        for news in news_list.data.get('news', []):
            results.append({
                'title': news.headline,
                'date': news.created_at.strftime('%Y-%m-%d'),
                'body': news.summary
            })
        return results
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []

def format_news_for_prompt(symbol: str, news: list) -> str:
    if not news:
        return f"No recent news found for {symbol}."
    
    formatted = f"Recent News for {symbol}:\n"
    for item in news:
        formatted += f"- {item['title']} ({item['date']}): {item['body']}\n"
    return formatted

def get_social_sentiment(symbol: str, max_results=3):
    """
    Alpaca does not have a dedicated Social Sentiment API.
    We use Alpaca News API as a proxy for sentiment analysis.
    """
    if not news_client:
        return []

    try:
        # Re-using News API as requested to stay within Alpaca ecosystem
        request_params = NewsRequest(
            symbols=symbol,
            limit=max_results,
            sort="desc"
        )
        news_list = news_client.get_news(request_params)
        
        results = []
        for news in news_list.data.get('news', []):
            results.append({
                'title': f"Alpaca News ({news.source}): {news.headline}",
                'date': news.created_at.strftime('%Y-%m-%d'),
                'body': news.summary
            })
        return results
    except Exception as e:
        print(f"Error fetching Alpaca sentiment news for {symbol}: {e}")
        return []

def format_sentiment_for_prompt(symbol: str, sentiment: list) -> str:
    if not sentiment:
        return ""
    
    formatted = f"Social Sentiment for {symbol}:\n"
    for item in sentiment:
        formatted += f"- {item['title']}: {item['body']}\n"
    return formatted


def _cache_get(cache_key: str) -> list[dict[str, Any]] | None:
    cached = _WEB_CACHE.get(cache_key)
    if not cached:
        return None

    expires_at, payload = cached
    if time.time() >= expires_at:
        _WEB_CACHE.pop(cache_key, None)
        return None

    return payload


def _cache_set(cache_key: str, payload: list[dict[str, Any]]) -> None:
    _WEB_CACHE[cache_key] = (time.time() + _WEB_TTL_SECONDS, payload)


def _lmstudio_playwright_search(*, query: str, max_results: int, days: int) -> list[dict[str, Any]]:
    cache_key = f"{query}|{max_results}|{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    local_url = os.getenv("LOCAL_LLM_URL", "http://host.docker.internal:1234/v1")
    # Clean url to root domain since native API uses /v1/responses or /api/v1/responses
    base_url = local_url.replace("/chat/completions", "").replace("/v1", "").rstrip("/")
    
    # Try /v1/responses (Most common)
    native_url = f"{base_url}/v1/responses"
    
    prompt = (
        f"Search the web for the following query: '{query}'\n\n"
        f"Analyze results from the last {days} days. Return exactly {max_results} updates. "
        "Output ONLY a valid JSON array of objects with keys: 'title', 'url', 'content', 'published_date', 'domain'."
    )
    
    model_to_use = get_active_local_model_sync()
    # LM Studio Responses API often expects 'input' instead of 'messages' for tool/plugin routing.
    # The identifier should match the key in your LM Studio mcp.json (which is "playwright").
    integration_id = "playwright" 
    
    payload = {
        "model": model_to_use,
        "input": prompt,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "integrations": [integration_id, f"mcp/{integration_id}"],
        "temperature": 0.0,
        "context_length": 20480
    }
 
    try:
        headers = {"Content-Type": "application/json"}
        if os.getenv("LM_STUDIO_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('LM_STUDIO_API_KEY')}"
            
        print(f"Calling LM Studio native responses API: {native_url} with integrations: mcp/playwright")
        response = requests.post(native_url, json=payload, headers=headers, timeout=_WEB_TIMEOUT_SECONDS)
        
        # If /v1/responses fails with 404 or "method not allowed", try /api/v1/responses
        if response.status_code in [404, 405]:
            alt_url = f"{base_url}/api/v1/responses"
            print(f"Retrying with alternate API: {alt_url}")
            response = requests.post(alt_url, json=payload, headers=headers, timeout=_WEB_TIMEOUT_SECONDS)

        response.raise_for_status()
        
        # /v1/responses returns choices like chat completions
        resp_data = response.json()
        raw_content = ""
        if "choices" in resp_data:
            raw_content = resp_data["choices"][0]["message"]["content"]
        elif "output" in resp_data:
            for out in resp_data["output"]:
                if out.get("type") == "message":
                    content_val = out.get("content", "")
                    if isinstance(content_val, list):
                        for c in content_val:
                            if isinstance(c, dict) and "text" in c:
                                raw_content += str(c["text"])
                            else:
                                raw_content += str(c)
                    else:
                        raw_content += str(content_val)
        else:
            raw_content = str(resp_data)
        
        import json
        import re
        
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw_content, re.DOTALL)
        if match:
            results = json.loads(match.group(0))
        else:
            results = json.loads(raw_content)
            
        normalized: list[dict[str, Any]] = []
        for item in results:
            url_str = item.get("url", "")
            domain = item.get("domain", "")
            if url_str and not domain:
                domain = urlparse(url_str).netloc.replace("www.", "")
            normalized.append({
                "title": item.get("title", "Untitled"),
                "content": item.get("content", ""),
                "url": url_str,
                "published_date": item.get("published_date", ""),
                "score": 0.9,
                "domain": domain or "unknown",
            })

        _cache_set(cache_key, normalized)
        return normalized

    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Permission denied" in error_msg:
            print(f"WEB RESEARCH BLOCKED: LM Studio Plugins/MCP are disabled or denied (403).")
            print("To fix: Enable 'Allow Plugins via API' in LM Studio Server settings.")
        else:
            print(f"LM Studio Playwright MCP search failed for query '{query}': {e}")
        return []


def get_macro_web_research(max_results: int = 6, days: int = 2) -> list[dict[str, Any]]:
    query = (
        "US stock market macro catalysts today: fed policy, treasury yields, inflation, "
        "earnings surprises, risk-on risk-off sentiment"
    )
    return _lmstudio_playwright_search(query=query, max_results=max_results, days=days)


def get_ticker_web_research(symbol: str, max_results: int = 5, days: int = 3) -> list[dict[str, Any]]:
    query = (
        f"{symbol} stock latest catalyst: earnings guidance, analyst upgrades downgrades, "
        "SEC filings, product launches, litigation, outlook"
    )
    return _lmstudio_playwright_search(query=query, max_results=max_results, days=days)


def format_web_research_for_prompt(scope: str, results: list[dict[str, Any]], max_items: int = 6) -> str:
    if not results:
        return ""

    lines = [f"Web Research ({scope}):"]
    for item in results[:max_items]:
        content = (item.get("content") or "").strip().replace("\n", " ")
        if len(content) > 220:
            content = content[:220] + "..."
        lines.append(
            "- "
            f"[{item.get('domain', 'source')}] score={item.get('score', 0)} "
            f"date={item.get('published_date', 'n/a')} | {item.get('title', 'Untitled')} | {content}"
        )
    return "\n".join(lines)


def shortlist_candidates_for_web_research(candidates: list[str], max_count: int = 3) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for symbol in candidates:
        key = symbol.strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
        if len(deduped) >= max_count:
            break
    return deduped
def get_market_data(symbols: list[str]) -> str:
    """
    Fetches the latest bars/snapshots for a list of symbols and returns a summary string.
    """
    from alpaca.data.requests import StockSnapshotRequest
    from core.clients import data_client
    
    if not data_client:
        return "Market data client not initialized."
    
    summary = ""
    try:
        request_params = StockSnapshotRequest(symbol_or_symbols=symbols)
        snapshots = data_client.get_stock_snapshot(request_params)
        
        for s in symbols:
            if s in snapshots:
                snap = snapshots[s]
                price = snap.latest_trade.price
                prev_close = snap.previous_daily_bar.close
                change = ((price - prev_close) / prev_close) * 100 if prev_close else 0
                
                bid = snap.latest_quote.bid_price
                ask = snap.latest_quote.ask_price
                spread = ask - bid if ask and bid else 0
                
                vol = snap.daily_bar.volume
                
                summary += f"{s}: Price={price}, Chg={round(change, 2)}%, Spread={round(spread, 3)}, Vol={vol} | "
            else:
                summary += f"{s}: no data | "
    except Exception as e:
        summary = f"Error fetching market data: {e}"
        
    return summary

def get_rsi(symbol: str, window: int = 14) -> float:
    """
    Calculates the 14-period RSI (Relative Strength Index) for a given symbol.
    """
    from core.clients import data_client
    if not data_client:
        return 50.0
        
    try:
        # Get 30 bars to ensure we have enough data for a 14-period RSI
        end = datetime.now()
        start = end - timedelta(days=5)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=BarTimeframe.Hour,
            start=start,
            end=end,
            feed=DataFeed.IEX
        )
        
        bars = data_client.get_stock_bars(request_params)
        df = bars.df
        
        if df.empty or len(df) < window:
            return 50.0
            
        # Standard RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1+rs))
        
        last_rsi = rsi.iloc[-1]
        return round(float(last_rsi), 2) if not (rsi.isna().iloc[-1]) else 50.0
    except Exception as e:
        print(f"Error calculating RSI for {symbol}: {e}")
        return 50.0
