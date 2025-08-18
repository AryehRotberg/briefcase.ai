"""
Database utilities for PostgreSQL connection and operations.
"""
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import POSTGRES_URL
from .base import Base


try:
    engine = create_engine(POSTGRES_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
except Exception as e:
    print(f"Failed to initialize database: {e}")
    raise

@contextmanager
def get_database_session():
    """
    Generator function to provide database session with proper logging and cleanup.
    
    Yields:
        Session: SQLAlchemy database session
        
    Raises:
        Exception: Database connection or session errors
    """
    session = SessionLocal()
    
    try:
        yield session
        
    except Exception as e:
        print(f"Error in database session: {e}")
        print("Rolling back database session due to error")
        session.rollback()
        raise
        
    finally:
        session.close()
