import json
import os

import numpy as np
import pandas as pd
import psycopg2


def connect():
    """
    Connects to the database
    :return: connection
    """
    return psycopg2.connect(os.environ['DATABASE_URL'])


def create_table(connection) -> None:
    """ Create table 'inference_logs' in database """
    with connection.cursor() as cur:
        cur.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            id serial PRIMARY KEY,
            inputs varchar(2000),
            outputs varchar(2000)
        );
        ''')


def log_inference(connection, features_df: pd.DataFrame, inference: np.ndarray) -> None:
    """Inserts the predictions into the inference_logs table"""
    records = features_df.to_dict(orient="records")
    with connection.cursor() as cur:
        for rec, inf in zip(records, inference):
            cur.execute(
                "INSERT INTO inference_logs(inputs, outputs) VALUES(%s, %s);",
                (json.dumps(rec), float(inf))
            )


def get_inferences(connection, limit: int) -> list:
    """
    Retrieves the last 10 records from the heroku database
    :return: 10 last records
    """
    with connection.cursor() as cur:
        cur.execute(
            "SELECT * FROM inference_logs ORDER BY id DESC LIMIT %s", (limit,))

        return cur.fetchall()
