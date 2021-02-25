from typing import Dict

from flask import jsonify
from werkzeug.wrappers import Response


class InvalidUsage(Exception):
    """Custom Exception to be raised on invalid user input.

    Taken from https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
    """
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self) -> Dict[str, str]:
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error: Exception) -> Response:
    """Function to handle invalid usage exceptions raised by code.

    Taken from https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response
