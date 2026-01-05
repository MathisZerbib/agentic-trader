import os
from dotenv import load_dotenv
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import NewsRequest, MostActivesRequest
from alpaca.data.enums import MostActivesBy

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

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
