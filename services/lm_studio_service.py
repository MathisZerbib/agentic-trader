import requests
from core.config import settings
from core.llm_settings import LLM_SETTINGS

class LMStudioService:
    def _get_base_url(self):
        # Use LLM_SETTINGS["local_url"] if available, fallback to settings
        # This ensures we use the URL the user might have updated in the frontend
        raw_url = LLM_SETTINGS.get("local_url") or settings.LOCAL_LLM_URL
        url = raw_url.strip().rstrip('/')
        
        # Sequentially strip common suffixes to get the raw base URL
        suffixes = ['/chat/completions', '/v1', '/models']
        changed = True
        while changed:
            changed = False
            for s in suffixes:
                if url.endswith(s):
                    url = url[:-len(s)].rstrip('/')
                    changed = True
                    break
        return url

    def get_headers(self):
        headers = {
            "Content-Type": "application/json"
        }
        # Try both env variants
        token = settings.LM_STUDIO_API_TOKEN
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def load_model(self, model_config: dict):
        base = self._get_base_url()
        url = f"{base}/api/v1/models/load"
        response = requests.post(url, json=model_config, headers=self.get_headers())
        response.raise_for_status()
        return response.json()

    def unload_model(self, instance_id: str):
        base = self._get_base_url()
        url = f"{base}/api/v1/models/unload"
        payload = {"instance_id": instance_id}
        response = requests.post(url, json=payload, headers=self.get_headers())
        response.raise_for_status()
        return response.json()

    def list_models_available(self):
        base = self._get_base_url()
        url = f"{base}/v1/models"
        response = requests.get(url, headers=self.get_headers())
        response.raise_for_status()
        return response.json()

lm_studio_service = LMStudioService()
