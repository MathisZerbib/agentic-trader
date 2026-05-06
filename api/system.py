
from core.clients import bot_state, grok_client
from core.config import settings
from core.llm_settings import LLM_SETTINGS
import agents.agents as agents
from agents.orchestrator import autonomous_cycle
from services.state import trigger_state_broadcast
import json
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models
from schemas import LLMSettingsUpdateRequest, LoadModelRequest


import logging
import httpx

logger = logging.getLogger(__name__)

def _sync_local_llm_settings():
    from core.llm_settings import LLM_SETTINGS, SETTINGS_FILE
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(LLM_SETTINGS, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to sync LLM settings to disk: {e}")

def _normalized_local_url(url: str) -> str:
    if not url: return ""
    u = url.strip().rstrip('/')
    while True:
        if u.endswith('/v1'): u = u[:-3]
        elif u.endswith('/chat/completions'): u = u[:-17]
        elif u.endswith('/models'): u = u[:-7]
        else: break
    return u + '/v1'

def _fetch_local_llm_models(local_url: str):
    if not local_url: return []
    normalized = _normalized_local_url(local_url).strip()
    if not normalized: return []
    headers = {"Authorization": "Bearer sk-lm-aAxaxZte:mOc432tNRd7CWCOh57g3"}
    try:
        r = httpx.get(f"{normalized}/models", timeout=2.0, headers=headers)
        if r.status_code == 200:
            return [m.get("id") for m in r.json().get("data", [])]
    except Exception: pass
    return []

def _active_engine_label() -> str:
    if LLM_SETTINGS["provider"] == "local":
        return f"Local ({LLM_SETTINGS['local_model']})"
    if agents.USE_LOCAL_FALLBACK:
        return f"Local Fallback ({LLM_SETTINGS['local_model']})"
    return f"Grok API ({LLM_SETTINGS['grok_model']})"

router = APIRouter()


@router.post("/bot/start")
async def start_bot():
    bot_state['BOT_ACTIVE'] = True
    await trigger_state_broadcast()
    return {"status": "Bot started", "bot_active": bot_state['BOT_ACTIVE']}

@router.post("/bot/stop")
async def stop_bot():
    bot_state['BOT_ACTIVE'] = False
    await trigger_state_broadcast()
    return {"status": "Bot stopped", "bot_active": bot_state['BOT_ACTIVE']}

@router.post("/bot/lock")
async def lock_trading():
    bot_state['TRADING_LOCKED'] = True
    await trigger_state_broadcast()
    return {"status": "Trading locked", "trading_locked": True}

@router.post("/bot/unlock")
async def unlock_trading():
    bot_state['TRADING_LOCKED'] = False
    await trigger_state_broadcast()
    return {"status": "Trading unlocked", "trading_locked": False}


@router.post("/run-agent")
async def run_agent(db: Session = Depends(get_db)):
    # Run a single cycle forcefully
    try:
        await autonomous_cycle(db, force=True)
        return {"status": "Manual agent cycle triggered"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@router.get("/settings/llm")
def get_llm_settings():
    return {
        "provider": LLM_SETTINGS["provider"],
        "grok_model": LLM_SETTINGS["grok_model"],
        "local_model": LLM_SETTINGS["local_model"],
        "local_url": _normalized_local_url(LLM_SETTINGS["local_url"]),
        "recommended_grok_model": settings.DEFAULT_GROK_MODEL,
        "fallback_active": agents.USE_LOCAL_FALLBACK,
        "active_engine": _active_engine_label(),
        "openrouter_available": bool(grok_client),
        "position_monitor_interval_seconds": settings.POSITION_MONITOR_INTERVAL_SECONDS,
        "trading_locked": bot_state.get("TRADING_LOCKED", False),
        "take_profit_percentage": LLM_SETTINGS.get("take_profit_percentage", settings.TAKE_PROFIT_PERCENTAGE),
        "stop_loss_percentage": LLM_SETTINGS.get("stop_loss_percentage", settings.STOP_LOSS_PERCENTAGE),
        "daily_drawdown_threshold": LLM_SETTINGS.get("daily_drawdown_threshold", settings.DAILY_DRAWDOWN_THRESHOLD),
        "web_research_enabled": LLM_SETTINGS.get("web_research_enabled", settings.WEB_RESEARCH_ENABLED),
        "web_research_max_tickers": LLM_SETTINGS.get("web_research_max_tickers", settings.WEB_RESEARCH_MAX_TICKERS),
        "web_research_days": LLM_SETTINGS.get("web_research_days", settings.WEB_RESEARCH_DAYS),
    }


@router.get("/settings/llm/models")
def get_llm_models(local_url: Optional[str] = None):
    url = local_url.strip() if local_url and local_url.strip() else LLM_SETTINGS["local_url"]
    local_models = _fetch_local_llm_models(url)
    return {
        "local_models": [m if isinstance(m, str) else m.get("key", m.get("id")) for m in local_models],
        "local_models_detail": local_models,
        "recommended_grok_model": settings.DEFAULT_GROK_MODEL,
        "local_url": _normalized_local_url(url),
    }


@router.post("/settings/llm")
async def update_llm_settings(payload: LLMSettingsUpdateRequest):
    provider = payload.provider.lower().strip()
    if provider not in {"grok", "local"}:
        raise HTTPException(status_code=400, detail="provider must be 'grok' or 'local'")

    if provider == "grok" and not grok_client:
        raise HTTPException(status_code=400, detail="Grok provider is not configured (set XAI_API_KEY or OPENROUTER_API_KEY)")

    if payload.grok_model and payload.grok_model.strip():
        LLM_SETTINGS["grok_model"] = payload.grok_model.strip()

    if payload.local_model and payload.local_model.strip():
        LLM_SETTINGS["local_model"] = payload.local_model.strip()

    if payload.local_url and payload.local_url.strip():
        LLM_SETTINGS["local_url"] = payload.local_url.strip()

    LLM_SETTINGS["provider"] = provider
    agents.USE_LOCAL_FALLBACK = provider == "local"
    
    if payload.position_monitor_interval_seconds:
        interval = max(30, payload.position_monitor_interval_seconds) # Min 30s
        settings.POSITION_MONITOR_INTERVAL_SECONDS = interval
        LLM_SETTINGS["position_monitor_interval_seconds"] = interval
        
        # Reschedule the job
        from main import scheduler
        try:
            scheduler.reschedule_job(
                'position_monitor', 
                trigger='interval', 
                seconds=interval
            )
            print(f"Rescheduled position_monitor to {interval}s")
        except Exception as e:
            print(f"Failed to reschedule job: {e}")

    if payload.take_profit_percentage is not None:
        LLM_SETTINGS["take_profit_percentage"] = payload.take_profit_percentage
    if payload.stop_loss_percentage is not None:
        LLM_SETTINGS["stop_loss_percentage"] = payload.stop_loss_percentage
    if payload.daily_drawdown_threshold is not None:
        LLM_SETTINGS["daily_drawdown_threshold"] = payload.daily_drawdown_threshold
    if payload.web_research_enabled is not None:
        LLM_SETTINGS["web_research_enabled"] = payload.web_research_enabled
    if payload.web_research_max_tickers is not None:
        LLM_SETTINGS["web_research_max_tickers"] = payload.web_research_max_tickers
    if payload.web_research_days is not None:
        LLM_SETTINGS["web_research_days"] = payload.web_research_days

    _sync_local_llm_settings()
    await trigger_state_broadcast()

    return {
        "status": "LLM settings updated",
        **get_llm_settings(),
    }


@router.post("/settings/llm/load-model")
async def load_local_model(payload: LoadModelRequest):
    """
    Proxies a model load request to LM Studio.
    """
    base_url = _normalized_local_url(LLM_SETTINGS["local_url"])
    load_url = f"{base_url}/models/load"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-lm-aAxaxZte:mOc432tNRd7CWCOh57g3"
    }
    
    body = {
        "model": payload.model,
        "context_length": payload.context_length,
        "flash_attention": payload.flash_attention,
        "echo_load_config": True
    }
    
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(load_url, json=body, headers=headers, timeout=120.0)
            if r.status_code != 200:
                logger.error(f"LM Studio load failed: {r.status_code} - {r.text}")
                try:
                    err_detail = r.json()
                except:
                    err_detail = r.text
                raise HTTPException(status_code=r.status_code, detail=err_detail)
            
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Could not connect to LM Studio")
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

