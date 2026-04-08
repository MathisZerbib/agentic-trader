
from core.clients import bot_state, grok_client
from core.config import settings
from core.llm_settings import LLM_SETTINGS
import agents.agents as agents
from services.state import trigger_state_broadcast
import json
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models
from schemas import LLMSettingsUpdateRequest


import logging
import httpx

logger = logging.getLogger(__name__)

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
def update_llm_settings(payload: LLMSettingsUpdateRequest):
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
    _sync_local_llm_settings()

    return {
        "status": "LLM settings updated",
        **get_llm_settings(),
    }

