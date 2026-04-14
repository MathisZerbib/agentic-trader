from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import uvicorn
import models
from database import engine

from core.config import settings
from api.system import router as system_router
from api.portfolio import router as portfolio_router
from api.ws import router as ws_router

# Setup DB models (will move to alembic later ideally)
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Grok Trading Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Add the background scheduled job here, linking to agents/orchestrator.py
from agents.orchestrator import autonomous_cycle
from services.state import scheduled_broadcast

scheduler = AsyncIOScheduler()
scheduler.add_job(scheduled_broadcast, 'interval', seconds=10)
# scheduler.add_job(autonomous_cycle, 'interval', minutes=30)
scheduler.add_job(autonomous_cycle, 'cron', day_of_week='mon-fri', hour=9, minute=30, timezone='America/New_York')

@app.on_event("startup")
def start_scheduler():
    scheduler.start()

# Include APIRouters
app.include_router(system_router)
app.include_router(portfolio_router)
app.include_router(ws_router)

from services.state import get_current_state

@app.get("/")
def read_root():
    state = get_current_state(None)
    # The frontend expects a flat dictionary for backward compatibility with root fetch
    # get_current_state returns a dict with "type": "state", "bot_active": ..., etc.
    return state

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
