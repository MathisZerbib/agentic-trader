from core.config import settings

LLM_SETTINGS = {
    "provider": settings.LOCAL_LLM_MODEL if settings.LOCAL_LLM_MODEL else "local",
    "grok_model": settings.DEFAULT_GROK_MODEL,
    "local_model": settings.LOCAL_LLM_MODEL,
    "local_url": settings.LOCAL_LLM_URL,
}
