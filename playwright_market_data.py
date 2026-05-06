import requests
import json
import os
from dotenv import load_dotenv

# Load .env to get the API key and model name
load_dotenv()

def playwright_market_data(stock_symbol: str):
    local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234")
    # If in docker, localhost needs replaced by host.docker.internal usually
    # but we'll assume the user provides the correct LOCAL_LLM_URL.
    # To be safe, if we are in Docker (often indicated by /.dockerenv), we might need fallback.
    
    url = f"{local_url}/api/v1/chat"
    if "/v1" in local_url and not local_url.endswith("/v1"):
        # If user provided something like http://host:1234/v1, make sure we use /api/v1/chat
        base = local_url.split("/v1")[0]
        url = f"{base}/api/v1/chat"
    
    api_key = os.getenv("LM_STUDIO_API_KEY")
    # Try to get active model or fallback to hardcoded
    try:
        from services.llm_utils import get_active_local_model_sync
        model = get_active_local_model_sync()
    except ImportError:
        model = "gemma-4-e2b-it"

    combined_prompt = f"""SYSTEM INSTRUCTIONS: You are an autonomous web-browsing agent using the Playwright tool. Follow these steps strictly:
STEP 1: Navigate to https://finance.yahoo.com/quote/{stock_symbol} using the `browser_navigate` tool.
STEP 2: Because you are in France, Yahoo redirects to a GDPR consent page. You MUST use the `browser_evaluate` tool to execute this exact JavaScript string: () => {{ const btn = document.querySelector('button.reject-all'); if(btn) btn.click(); return "clicked"; }}
STEP 3: After executing that script, use `browser_evaluate` again to extract all the financial metrics and analyst data. Execute this exact JavaScript string: 
() => {{
    const getVal = (field) => {{ 
        const el = document.querySelector('fin-streamer[data-field="' + field + '"]'); 
        return el ? el.innerText : "N/A"; 
    }};
    const getNews = () => {{
        const el = document.querySelector('h3.clamp');
        return el ? el.innerText : "N/A";
    }};
    
    const data = {{
        Metrics: {{
            "Price": getVal('regularMarketPrice'),
            "Change %": getVal('regularMarketChangePercent'),
            "Previous Close": getVal('regularMarketPreviousClose'),
            "Open": getVal('regularMarketOpen'),
            "Day Range": getVal('regularMarketDayRange'),
            "Volume": getVal('regularMarketVolume'),
            "Market Cap": getVal('marketCap'),
            "P/E Ratio (Trailing)": getVal('trailingPE'),
            "Target Price": getVal('targetMeanPrice')
        }},
        TopNewsHeadline: getNews(),
        AnalystInsights: {{}}
    }};

    const topCard = document.querySelector('[data-testid="top-analyst-card"]');
    if(topCard) {{
        data.AnalystInsights.TopAnalystName = topCard.querySelector('.score div:first-child') ? topCard.querySelector('.score div:first-child').innerText : 'N/A';
        data.AnalystInsights.TopAnalystScore = topCard.querySelector('.score div:last-child') ? topCard.querySelector('.score div:last-child').innerText : 'N/A';
        data.AnalystInsights.TopAnalystRating = topCard.querySelector('[data-testid="status-tg"]') ? topCard.querySelector('[data-testid="status-tg"]').innerText : 'N/A';
    }}

    const ptCard = document.querySelector('[data-testid="analyst-price-target-card"]');
    if(ptCard) {{
        data.AnalystInsights.TargetLow = ptCard.querySelector('.lowLabel .price') ? ptCard.querySelector('.lowLabel .price').innerText : 'N/A';
        data.AnalystInsights.TargetAvg = ptCard.querySelector('.average .price') ? ptCard.querySelector('.average .price').innerText : 'N/A';
        data.AnalystInsights.TargetHigh = ptCard.querySelector('.highLabel .price') ? ptCard.querySelector('.highLabel .price').innerText : 'N/A';
    }}
    
    return JSON.stringify(data);
}}
STEP 4: Read the JSON data returned from the browser. You MUST output the metrics as a Markdown table with the columns "Metric" and "Value". Below the table, provide the Top News Headline and a formatted summary of the Analyst Insights.

### ANTI-HALLUCINATION GUARDRAIL
If you cannot access the page, the tool fails, or you are redirected to a page without the requested metrics, you MUST output exactly: "RESEARCH_UNAVAILABLE: [Reason]".
NEVER simulate financial data. NEVER invent news. NEVER provide "representative" data. 
If a specific metric is missing but others are found, use "N/A" for that metric only.

USER REQUEST: Give me the metric table, news, and analyst breakdown for {stock_symbol} based on the current Yahoo Finance page."""

    payload = {
        "model": model,
        "input": combined_prompt,
        "integrations": ["mcp/playwright"],
        "context_length": 8000,
        "temperature": 0.1 
    }

    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"🚀 Launching Ultimate Playwright Scraper for {stock_symbol}...")
    print("⏳ Fetching metrics table, news, and analyst data...")
    
    import time
    max_retries = 2
    retry_delay = 5
    
    # Use a long timeout for browsing (3 minutes)
    timeout = 180 

    for attempt in range(max_retries):
        try:
            print(f"📡 [Attempt {attempt+1}/{max_retries}] Sending request to LM Studio...")
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                output = data.get('output', [])
                if not output:
                    print(f"❓ Response received but no 'output' found: {json.dumps(data, indent=2)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return "No output received after multiple attempts."

                for item in output:
                    if item.get('type') == 'reasoning':
                        # print(f"🧠 Reasoning: {item.get('content')}\n")
                        pass
                    elif item.get('type') == 'tool_call':
                        print(f"🛠️  TOOL TRIGGERED: {item.get('tool')}")
                    elif item.get('type') == 'message':
                        content = item.get('content')
                        full_text = ""
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and 'text' in c:
                                    full_text += c['text']
                        else:
                            full_text = str(content)
                        
                        if full_text:
                            print(f"✅ Playwright Research Completed for {stock_symbol}")
                            return full_text
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return "No data found in final response."
            else:
                print(f"⚠️ Error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return f"Error {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout on attempt {attempt+1} for {stock_symbol}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return f"Timeout Error: Research for {stock_symbol} took too long (> {timeout}s)"
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection Error on attempt {attempt+1} for {stock_symbol}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "Connection Error: Is LM Studio running on port 1234?"
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return f"Unexpected Error: {e}"
    
    return "Failed to fetch research after retries."
        