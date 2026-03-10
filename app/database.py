from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import urllib

# Build connection string for SQLAlchemy
params = urllib.parse.quote_plus(settings.DB_CONNECTION)
DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,          # set True to see SQL queries in terminal
    pool_pre_ping=True,  # checks connection is alive before using
    pool_size=5,         # max 5 connections in pool
    max_overflow=10      # up to 10 extra connections if needed
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for all models
Base = declarative_base()

# Dependency — used in every API endpoint
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Test connection
def test_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ Database connected successfully")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
