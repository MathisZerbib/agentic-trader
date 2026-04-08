import re

with open("backend/main.py", "r") as f:
    content = f.read()

imports = """
from api.system import router as system_router
from api.portfolio import router as portfolio_router
"""
if "from api.system" not in content:
    content = content.replace("from fastapi import FastAPI, HTTPException", "from fastapi import FastAPI, HTTPException\n" + imports)

app_includes = """
app.include_router(system_router)
app.include_router(portfolio_router)
"""

if "app.include_router(system_router)" not in content:
    content = content.replace('app = FastAPI(title="Grok Trading Bot")', 'app = FastAPI(title="Grok Trading Bot")\n' + app_includes)

with open("backend/main.py", "w") as f:
    f.write(content)
