import numpy as np
import pandas as pd
from sqlalchemy.engine import Connection

from src.database.models import InferenceLog


def log_inference(connection: Connection, features: pd.DataFrame, inferred: np.ndarray):
    """Log features used and inference made to a database."""
    df = features.copy()
    df.loc[:, "inferred_monthly_rent"] = inferred
    df.to_sql(
        name=InferenceLog.__tablename__,
        con=connection,
        schema=InferenceLog.__table__.schema,
        if_exists="append",
        index=False,
    )
