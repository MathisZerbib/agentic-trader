import re
import os

ORCHESTRATOR_PATH = "backend/agents/orchestrator.py"

if not os.path.exists(ORCHESTRATOR_PATH):
    print(f"Error: {ORCHESTRATOR_PATH} not found.")
    exit(1)

with open(ORCHESTRATOR_PATH, "r") as f:
    content = f.read()

# Replacements mapping
replacements = {
    "settings.TAKE_PROFIT_PERCENTAGE": 'LLM_SETTINGS.get("take_profit_percentage", 0.05)',
    "settings.STOP_LOSS_PERCENTAGE": 'LLM_SETTINGS.get("stop_loss_percentage", -0.03)',
    "settings.DAILY_DRAWDOWN_THRESHOLD": 'LLM_SETTINGS.get("daily_drawdown_threshold", -0.03)',
    "settings.WEB_RESEARCH_ENABLED": 'LLM_SETTINGS.get("web_research_enabled", True)',
    "settings.WEB_RESEARCH_MAX_TICKERS": 'LLM_SETTINGS.get("web_research_max_tickers", 3)',
    "settings.WEB_RESEARCH_DAYS": 'LLM_SETTINGS.get("web_research_days", 3)',
    "settings.WEB_RESEARCH_MACRO_MAX_RESULTS": 'LLM_SETTINGS.get("web_research_macro_max_results", 6)',
    "settings.WEB_RESEARCH_TICKER_MAX_RESULTS": 'LLM_SETTINGS.get("web_research_ticker_max_results", 5)',
}

new_content = content
for old, new in replacements.items():
    new_content = new_content.replace(old, new)

if new_content != content:
    with open(ORCHESTRATOR_PATH, "w") as f:
        f.write(new_content)
    print(f"Successfully patched {ORCHESTRATOR_PATH}")
else:
    print(f"No changes needed for {ORCHESTRATOR_PATH}")
