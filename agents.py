import json
from datetime import datetime
import asyncio
from agent_prompts import (
    STRATEGIST_SYSTEM_PROMPT, STRATEGIST_TASK_TEMPLATE,
    ANALYST_SYSTEM_PROMPT, ANALYST_TASK_TEMPLATE,
    RISK_MANAGER_SYSTEM_PROMPT, RISK_MANAGER_TASK_TEMPLATE,
    SENTIMENT_SYSTEM_PROMPT, SENTIMENT_TASK_TEMPLATE,
    REGIME_ARBITER_SYSTEM_PROMPT, REGIME_ARBITER_TASK_TEMPLATE,
    ADVERSARIAL_SYSTEM_PROMPT, ADVERSARIAL_TASK_TEMPLATE,
    TRADE_REVIEWER_SYSTEM_PROMPT, TRADE_REVIEWER_TASK_TEMPLATE
)

class BaseAgent:
    def __init__(self, client, model="x-ai/grok-4.1-fast"):
        self.client = client
        self.model = model

    async def _call_llm(self, system_prompt, user_prompt):
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
            print(f"LLM Error: {e}")
            return None

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
