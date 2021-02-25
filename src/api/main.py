import json
import pickle

import flask
from flask import jsonify, request
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.api.exceptions import InvalidInput
from src.api.utils import app, get_db
import src.database
from src.models import encode_data

load_dotenv()

model = pickle.load(open("models/model.pck", "rb"))
encoder = pickle.load(open("models/encoder.pck", "rb"))
scaler = pickle.load(open("models/scaler.pck", "rb"))


with app.app_context():
    with get_db() as connection:
        src.database.create_table(connection)


@app.route("/", methods=["GET"])
def hello() -> str:
    """View to check if server's up and running."""
    return "Hello, world!"


def _prepare_for_inference(request_data: str) -> np.ndarray:
    """Transforms request data into format model can work with."""
    features_df = pd.DataFrame(json.loads(request_data)["features"])
    encoded, *_ = encode_data(features_df, encoder, scaler)
    return encoded, features_df


@app.route("/predict/", methods=["POST"])
def make_inference(
) -> str:
    """Makes inference and returns predicted value."""
    try:
        prepared_features, features_df = _prepare_for_inference(request.data)
    except Exception as e:
        raise
        raise InvalidInput("massage")

    result = model.predict(prepared_features)

    with get_db() as db_connection:
        src.database.log_inference(db_connection, features_df, result)

    return jsonify({"inference": result.tolist()})


@app.route("/inferences/", methods=["GET"])
def get_last_inferences() -> str:
    """Retrieves the last records from database

    Arguments:
    :limit: int. How many records to retrieve. Defaults to 10.
    """
    limit = request.args.get("limit", 10)
    with get_db() as db_connection:
        records = src.database.get_inferences(db_connection, limit)
    return json.dumps({"inferences": records})
