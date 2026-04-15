from fastapi import APIRouter, HTTPException
from schemas import LoadModelRequest, ModelUnloadRequest
from services.lm_studio_service import lm_studio_service

router = APIRouter(prefix="/api/v1/models", tags=["models"])

@router.post("/load")
async def load_model(request: LoadModelRequest):
    try:
        # Filter out None values to avoid sending them to LM Studio if they weren't provided
        payload = {k: v for k, v in request.dict().items() if v is not None}
        result = lm_studio_service.load_model(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unload")
async def unload_model(request: ModelUnloadRequest):
    try:
        result = lm_studio_service.unload_model(request.instance_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("")
async def list_models():
    """List loaded models (via standard /v1/models)"""
    try:
        result = lm_studio_service.list_models_available()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
