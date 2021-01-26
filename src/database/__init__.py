from contextlib import contextmanager
import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker


engine: Engine = create_engine(
        f"postgresql://{os.environ['DATABASE_USER']}:{os.environ['DATABASE_PASSWORD']}"
        f"@{os.environ['DATABASE_HOST']}:{os.environ['DATABASE_PORT']}"
        f"/{os.environ['DATABASE_NAME']}"
    )


def SessionLocal() -> Session:
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def get_connection() -> Connection:
    """Connects to the database.

    Right now this leaks connections, un
    """
    connection = engine.connect()
    return connection