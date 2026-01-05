from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
import datetime

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    qty = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    reason = Column(String)
    order_id = Column(String, nullable=True)
    status = Column(String, default="new")

class DailyEquity(Base):
    __tablename__ = "daily_equity"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.datetime.utcnow)
    equity = Column(Float)
    pnl = Column(Float)

class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    title = Column(String)
    content = Column(String)

