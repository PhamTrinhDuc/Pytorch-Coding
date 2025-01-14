from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()

engine = create_engine(url=DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        # Attempt to connect to the database
        conn = engine.connect()
        conn.close() 
        yield db 
    except Exception as e:
        raise Exception(f"No active database connection: {e}") 
    finally:
        db.close()