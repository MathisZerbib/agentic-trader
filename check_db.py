from database import SessionLocal
from models import Trade, AgentLog

def check_db():
    db = SessionLocal()
    try:
        print("--- Recent Trades ---")
        trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(5).all()
        for t in trades:
            print(f"{t.timestamp} - {t.side} {t.qty} {t.symbol} @ {t.price} ({t.status}) - {t.reason}")
        
        print("\n--- Recent Agent Logs ---")
        logs = db.query(AgentLog).order_by(AgentLog.timestamp.desc()).limit(5).all()
        for l in logs:
            print(f"{l.timestamp} - {l.title}: {l.content}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_db()
