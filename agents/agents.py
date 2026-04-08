import json
import os
from dotenv import load_dotenv
from datetime import datetime
import asyncio
from openai import AsyncOpenAI
import urllib.request

load_dotenv()
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "local-model")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL")
DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "x-ai/grok-4.1-fast")
LM_STUDIO_API_TOKEN = os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY")

# Circuit breaker flag for OpenRouter 402 errors
USE_LOCAL_FALLBACK = False

async def call_local_llm(system_prompt, user_prompt):
    # Prioritize configured URL and support both LM Studio API and OpenAI-compatible paths.
    seed_urls = [
        LOCAL_LLM_URL,
        "http://host.docker.internal:1234/v1/chat/completions",
        "http://localhost:1234/v1/chat/completions",
    ]
    chat_urls = []
    for raw_url in seed_urls:
        if not raw_url:
            continue

        clean_url = raw_url.rstrip("/")
        if clean_url.endswith("/chat/completions"):
            chat_urls.append(clean_url)
            chat_urls.append(clean_url.replace("/v1/chat/completions", "/api/v1/chat/completions"))
        elif clean_url.endswith("/v1"):
            root_url = clean_url[:-3]
            chat_urls.append(f"{root_url}/api/v1/chat/completions")
            chat_urls.append(f"{clean_url}/chat/completions")
        else:
            chat_urls.append(f"{clean_url}/api/v1/chat/completions")
            chat_urls.append(f"{clean_url}/v1/chat/completions")

    # Remove duplicates while preserving order.
    chat_urls = list(dict.fromkeys(chat_urls))

    async def _post_chat_completion(endpoint: str, payload: dict) -> str:
        def _do_request() -> str:
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(endpoint, data=body, method="POST")
            request.add_header("Content-Type", "application/json")
            if LM_STUDIO_API_TOKEN:
                request.add_header("Authorization", f"Bearer {LM_STUDIO_API_TOKEN}")

            with urllib.request.urlopen(request, timeout=30) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            choices = response_data.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                if isinstance(message, dict) and message.get("content"):
                    return message["content"]

            raise ValueError(f"Unexpected LM Studio response payload keys: {list(response_data.keys())}")

        return await asyncio.to_thread(_do_request)

    for endpoint in chat_urls:
        try:
            print(f"Calling Local LLM at {endpoint}...")

            # First try with json_object, but be ready to fallback to text.
            try:
                # Truncate inputs if they're too large for local models.
                if len(system_prompt) + len(user_prompt) > 12000:  # Rough char count for ~3-4k tokens
                    print("Truncating prompt for local model context limit...")
                    user_prompt = user_prompt[:8000] + "...(truncated)"

                content = await _post_chat_completion(
                    endpoint,
                    {
                        "model": LOCAL_LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "max_tokens": 768,
                    },
                )
            except Exception as e_format:
                if "response_format" in str(e_format) or "400" in str(e_format):
                    print(f"Local LLM issue (format/context), retrying with text format...")
                    # Aggressive truncation for fallback.
                    if len(system_prompt) + len(user_prompt) > 8000:
                        user_prompt = user_prompt[:5000] + "...(truncated)"

                    content = await _post_chat_completion(
                        endpoint,
                        {
                            "model": LOCAL_LLM_MODEL,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "max_tokens": 768,
                        },
                    )
                else:
                    raise e_format

            # Clean markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]
            
            # Clean double braces {{ ... }} sometimes produced by smaller models
            content = content.strip()
            if content.startswith("{{") and content.endswith("}}"):
                content = content.replace("{{", "{").replace("}}", "}")
            
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # 2. Try regex-like finder for { ... }
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    # Double brace cleanup for extracted string too
                    if json_str.startswith("{{") and json_str.endswith("}}"):
                        json_str = json_str.replace("{{", "{").replace("}}", "}")
                    return json.loads(json_str)
                return json.loads(content.strip()) # Last ditch effort
            except:
                raise ValueError(f"Could not extract JSON from text response: {content[:100]}...")

        except Exception as fallback_error:
            print(f"Fallback attempt to {endpoint} failed: {fallback_error}")
            continue
    
    print("All fallback attempts failed.")
    return {}

from agents.prompts import (
    STRATEGIST_SYSTEM_PROMPT, STRATEGIST_TASK_TEMPLATE,
    ANALYST_SYSTEM_PROMPT, ANALYST_TASK_TEMPLATE,
    RISK_MANAGER_SYSTEM_PROMPT, RISK_MANAGER_TASK_TEMPLATE,
    SENTIMENT_SYSTEM_PROMPT, SENTIMENT_TASK_TEMPLATE,
    REGIME_ARBITER_SYSTEM_PROMPT, REGIME_ARBITER_TASK_TEMPLATE,
    ADVERSARIAL_SYSTEM_PROMPT, ADVERSARIAL_TASK_TEMPLATE,
    TRADE_REVIEWER_SYSTEM_PROMPT, TRADE_REVIEWER_TASK_TEMPLATE
)

class BaseAgent:
    def __init__(self, client, model=DEFAULT_GROK_MODEL):
        self.client = client
        self.model = model

    async def _call_llm(self, system_prompt, user_prompt):
        global USE_LOCAL_FALLBACK
        
        if USE_LOCAL_FALLBACK:
            return await call_local_llm(system_prompt, user_prompt)
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
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
            # Check for 402 Insufficient credits or generic API errors
            error_msg = str(e).lower()
            if "402" in error_msg or "insufficient credits" in error_msg:
                print(f"LLM Error (402): {e}. PERMANENTLY Switching to fallback model ({LOCAL_LLM_MODEL}).")
                USE_LOCAL_FALLBACK = True
                return await call_local_llm(system_prompt, user_prompt)
            else:
                print(f"LLM Error: {e}")
                return {}

class Strategist(BaseAgent):
    async def get_regime(self, market_data, portfolio_state):
        prompt = STRATEGIST_TASK_TEMPLATE.format(
            vix=market_data.get('vix', 15.0),
            equity=portfolio_state.get('equity', 0.0),
            buying_power=portfolio_state.get('buying_power', 0.0),
            news_summary=market_data.get('news', "No major news.")
        )
        return await self._call_llm(STRATEGIST_SYSTEM_PROMPT, prompt)

class Analyst(BaseAgent):
    async def analyze_ticker(self, ticker, price_data):
        prompt = ANALYST_TASK_TEMPLATE.format(
            ticker=ticker,
            price=price_data['price'],
            rsi=price_data.get('rsi', 50),
            order_flow=price_data.get('order_flow', "Neutral"),
            market_data=price_data.get('market_context', "Trending"),
            sentiment_analysis=price_data.get('sentiment_analysis', "No sentiment data.")
        )
        return await self._call_llm(ANALYST_SYSTEM_PROMPT, prompt)

class SentimentAgent(BaseAgent):
    async def analyze_sentiment(self, ticker, news_headlines):
        prompt = SENTIMENT_TASK_TEMPLATE.format(
            ticker=ticker,
            news_headlines=news_headlines
        )
        return await self._call_llm(SENTIMENT_SYSTEM_PROMPT, prompt)

class RegimeArbiter(BaseAgent):
    async def determine_regime(self, market_snapshot, sentiment_summary, vix=20.0):
        prompt = REGIME_ARBITER_TASK_TEMPLATE.format(
            vix=vix,
            market_snapshot=market_snapshot,
            sentiment_summary=sentiment_summary
        )
        return await self._call_llm(REGIME_ARBITER_SYSTEM_PROMPT, prompt)

class AdversarialAgent(BaseAgent):
    async def challenge_trade(self, ticker, signal, thesis, price):
        prompt = ADVERSARIAL_TASK_TEMPLATE.format(
            ticker=ticker,
            signal=signal,
            thesis=thesis,
            price=price
        )
        return await self._call_llm(ADVERSARIAL_SYSTEM_PROMPT, prompt)

class RiskManager(BaseAgent):
    async def validate_trade(
        self,
        *,
        signal: str,
        conviction: float,
        ticker: str,
        requested_qty: float,
        entry_price: float,
        stop_price: float,
        current_regime: str,
        equity: float,
        buying_power: float,
        max_pos_size_pct: float = 10,
        max_total_exposure: float = 1.5,
    ):
        prompt = RISK_MANAGER_TASK_TEMPLATE.format(
            signal=signal,
            conviction=conviction,
            ticker=ticker,
            requested_qty=requested_qty,
            entry_price=entry_price,
            stop_price=stop_price,
            current_regime=current_regime,
            equity=equity,
            buying_power=buying_power,
        )
        sys_prompt = RISK_MANAGER_SYSTEM_PROMPT.format(
            max_pos_size_pct=max_pos_size_pct,
            max_total_exposure=max_total_exposure,
        )
        return await self._call_llm(sys_prompt, prompt)


class TradeReviewer(BaseAgent):
    async def review(self, *, as_of_date: str, trade_logs: str, evidence: str, market_risks: str):
        prompt = TRADE_REVIEWER_TASK_TEMPLATE.format(
            as_of_date=as_of_date,
            trade_logs=trade_logs,
            evidence=evidence,
            market_risks=market_risks,
        )
        return await self._call_llm(TRADE_REVIEWER_SYSTEM_PROMPT, prompt)

from agents.prompts import POSITION_MONITOR_SYSTEM_PROMPT, POSITION_MONITOR_TASK_TEMPLATE

class PositionMonitor(BaseAgent):
    async def monitor_position(self, pos_data, market_context):
        prompt = POSITION_MONITOR_TASK_TEMPLATE.format(
            symbol=pos_data["symbol"],
            qty=pos_data["qty"],
            current_price=pos_data["current_price"],
            avg_entry=pos_data["avg_entry"],
            unrealized_plpc=pos_data["unrealized_plpc"],
            tp_threshold=pos_data["tp_threshold"],
            sl_threshold=pos_data["sl_threshold"],
            market_context=market_context
        )
        return await self._call_llm(POSITION_MONITOR_SYSTEM_PROMPT, prompt)
