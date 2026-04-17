import re

with open("backend/services/market_data.py", "r") as f:
    content = f.read()

# Replace Tavily variables
content = re.sub(
    r'_TAVILY_SEARCH_URL = "https://api.tavily.com/search".*?_TAVILY_TIMEOUT_SECONDS = float\(os.getenv\("WEB_RESEARCH_TIMEOUT_SEC", "4"\)\)',
    r'''_LMSTUDIO_SEARCH_URL = os.getenv("LOCAL_LLM_URL", "http://host.docker.internal:1234/v1") + "/chat/completions"
_WEB_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_WEB_TTL_SECONDS = int(os.getenv("WEB_RESEARCH_TTL_SEC", "900"))
_WEB_TIMEOUT_SECONDS = float(os.getenv("WEB_RESEARCH_TIMEOUT_SEC", "120")) # extended for LM studio browsing''',
    content,
    flags=re.DOTALL
)

# Fix _cache_get and _cache_set to use _WEB_CACHE
content = content.replace("_TAVILY_CACHE", "_WEB_CACHE")
content = content.replace("_TAVILY_TTL_SECONDS", "_WEB_TTL_SECONDS")

# Replace _tavily_search definition
tavily_func = r'''def _tavily_search\(\*, query: str, max_results: int, days: int\) -> list\[dict\[str, Any\]\]:
    if not TAVILY_API_KEY:
        return \[\]

    cache_key = f"\{query\}\|\{max_results\}\|\{days\}"
    cached = _cache_get\(cache_key\)
    if cached is not None:
        return cached

    payload = \{
        "api_key": TAVILY_API_KEY,
        "query": query,
        "topic": "news",
        "search_depth": "advanced",
        "max_results": max_results,
        "include_raw_content": False,
        "include_images": False,
        "days": days,
    \}

    try:
        response = requests.post\(_TAVILY_SEARCH_URL, json=payload, timeout=_TAVILY_TIMEOUT_SECONDS\)
        response.raise_for_status\(\)
        body = response.json\(\)
    except Exception as e:
        print\(f"Tavily search failed for query '\{query\}': \{e\}"\)
        return \[\]

    normalized: list\[dict\[str, Any\]\] = \[\]
    for item in body.get\("results", \[\]\):
        url = item.get\("url", ""\)
        domain = urlparse\(url\).netloc.replace\("www.", ""\) if url else "unknown"
        normalized.append\(
            \{
                "title": item.get\("title", "Untitled"\),
                "content": item.get\("content", ""\),
                "url": url,
                "published_date": item.get\("published_date", ""\),
                "score": round\(float\(item.get\("score", 0.0\)\), 3\),
                "domain": domain,
            \}
        \)

    _cache_set\(cache_key, normalized\)
    return normalized'''

lmstudio_func = r'''def _lmstudio_playwright_search(*, query: str, max_results: int, days: int) -> list[dict[str, Any]]:
    # Use LM Studio's Playwright MCP by asking the local LLM to search
    cache_key = f"{query}|{max_results}|{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    prompt = (
        f"You have access to a Playwright MCP tool for web browsing.\n"
        f"Please search the web for the following query:\n"
        f"'{query}'\n\n"
        f"Analyze the search results for the last {days} days, and return exactly {max_results} relevant news articles or updates. "
        "You MUST return ONLY a valid JSON array containing objects with these exact keys: "
        "'title', 'url', 'content' (a brief summary), 'published_date', and 'domain'."
    )

    url = os.getenv("LOCAL_LLM_URL", "http://host.docker.internal:1234/v1")
    if not url.endswith("/v1"): url += "/v1"
    
    payload = {
        "model": get_active_local_model_sync(),
        "messages": [
            {"role": "system", "content": "You are a web research assistant. Use your Playwright MCP tool to browse the internet. Output strictly a JSON array."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(f"{url}/chat/completions", json=payload, timeout=_WEB_TIMEOUT_SECONDS)
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        
        import json
        import re
        
        # Try to extract the JSON array from the response
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
        print(f"LM Studio Playwright MCP search failed for query '{query}': {e}")
        return []'''

content = re.sub(tavily_func, lmstudio_func, content, flags=re.DOTALL)

content = content.replace("return _tavily_search(", "return _lmstudio_playwright_search(")

with open("backend/services/market_data.py", "w") as f:
    f.write(content)

print("Patched market_data.py")
