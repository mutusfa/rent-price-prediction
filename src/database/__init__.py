from contextlib import contextmanager
import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection


def Engine():
    """Creates an sqlalchemy engine for the database."""
    return create_engine(
        f"postgresql://{os.environ['DATABASE_USER']}:{os.environ['DATABASE_PASSWORD']}"
        f"@{os.environ['DATABASE_HOST']}:{os.environ['DATABASE_PORT']}"
        f"/{os.environ['DATABASE_NAME']}"
    )


def get_connection(engine) -> Connection:
    """Connects to the database.

    Right now this leaks connections, un
    """
    connection = engine.connect()
    return connection
