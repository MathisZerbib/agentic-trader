import os
import time
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import NewsRequest, MostActivesRequest
from alpaca.data.enums import MostActivesBy

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_TAVILY_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_TAVILY_TTL_SECONDS = int(os.getenv("WEB_RESEARCH_TTL_SEC", "900"))
_TAVILY_TIMEOUT_SECONDS = float(os.getenv("WEB_RESEARCH_TIMEOUT_SEC", "4"))

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
    cached = _TAVILY_CACHE.get(cache_key)
    if not cached:
        return None

    expires_at, payload = cached
    if time.time() >= expires_at:
        _TAVILY_CACHE.pop(cache_key, None)
        return None

    return payload


def _cache_set(cache_key: str, payload: list[dict[str, Any]]) -> None:
    _TAVILY_CACHE[cache_key] = (time.time() + _TAVILY_TTL_SECONDS, payload)


def _tavily_search(*, query: str, max_results: int, days: int) -> list[dict[str, Any]]:
    if not TAVILY_API_KEY:
        return []

    cache_key = f"{query}|{max_results}|{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "topic": "news",
        "search_depth": "advanced",
        "max_results": max_results,
        "include_raw_content": False,
        "include_images": False,
        "days": days,
    }

    try:
        response = requests.post(_TAVILY_SEARCH_URL, json=payload, timeout=_TAVILY_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
    except Exception as e:
        print(f"Tavily search failed for query '{query}': {e}")
        return []

    normalized: list[dict[str, Any]] = []
    for item in body.get("results", []):
        url = item.get("url", "")
        domain = urlparse(url).netloc.replace("www.", "") if url else "unknown"
        normalized.append(
            {
                "title": item.get("title", "Untitled"),
                "content": item.get("content", ""),
                "url": url,
                "published_date": item.get("published_date", ""),
                "score": round(float(item.get("score", 0.0)), 3),
                "domain": domain,
            }
        )

    _cache_set(cache_key, normalized)
    return normalized


def get_macro_web_research(max_results: int = 6, days: int = 2) -> list[dict[str, Any]]:
    query = (
        "US stock market macro catalysts today: fed policy, treasury yields, inflation, "
        "earnings surprises, risk-on risk-off sentiment"
    )
    return _tavily_search(query=query, max_results=max_results, days=days)


def get_ticker_web_research(symbol: str, max_results: int = 5, days: int = 3) -> list[dict[str, Any]]:
    query = (
        f"{symbol} stock latest catalyst: earnings guidance, analyst upgrades downgrades, "
        "SEC filings, product launches, litigation, outlook"
    )
    return _tavily_search(query=query, max_results=max_results, days=days)


def format_tavily_for_prompt(scope: str, results: list[dict[str, Any]], max_items: int = 6) -> str:
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
