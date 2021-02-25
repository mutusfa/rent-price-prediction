import flask
from flask import g

import src.database

app = flask.Flask(__name__)


def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.

    Taken from https://flask.palletsprojects.com/en/1.1.x/tutorial/database/
    """
    if 'db_connection' not in g:
        g.db_connection = src.database.connect()
    return g.db_connection


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if 'db_connection' in g:
        g.db_connection.close()
