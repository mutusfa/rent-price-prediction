import pytest

import src.database as database


@pytest.fixture
def engine():
    return database.engine


@pytest.fixture
def connection(engine):
    try:
        connection = engine.connect()
        yield connection
    finally:
        connection.close()


def test_connection(connection):
    assert connection.execute("SELECT 1")
