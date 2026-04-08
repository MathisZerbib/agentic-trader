# agentic-trader

## Optional Web Research (Tavily)

Add these variables to `backend/.env` to enable web-search evidence for regime and top ticker analysis:

```
TAVILY_API_KEY=your_tavily_key
WEB_RESEARCH_ENABLED=true
WEB_RESEARCH_MAX_TICKERS=3
WEB_RESEARCH_MACRO_MAX_RESULTS=6
WEB_RESEARCH_TICKER_MAX_RESULTS=5
WEB_RESEARCH_DAYS=3
WEB_RESEARCH_TIMEOUT_SEC=4
WEB_RESEARCH_TTL_SEC=900
```

Behavior:
- Runs one macro Tavily search each cycle.
- Runs ticker Tavily search only on a short list of top candidates.
- Uses timeout and in-memory cache to avoid slowing execution.
