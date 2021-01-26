import pickle
from typing import List

from fastapi import Depends, FastAPI
from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.engine import Connection

from src.models import encode_data
from src.database.repository import log_inference, get_inferences
from src.database import engine, get_connection, SessionLocal
import src.database.models

app = FastAPI()
model = pickle.load(open("models/model.pck", "rb"))
encoder = pickle.load(open("models/encoder.pck", "rb"))
scaler = pickle.load(open("models/scaler.pck", "rb"))

src.database.models.Base.metadata.create_all(bind=engine)

# Dependency
def get_session():
    """Starts a session and closes it after user is done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def hello():
    return "Hello, world!"


class FeaturesForRentInference(BaseModel):
    city: str
    district: str
    latitude: float
    longitude: float
    street: str
    floor_area_m2: float
    number_of_rooms: float
    floor: float
    number_of_floors: float
    build_year: float
    building_type: str
    heating_type: str
    equipment: str


class InferenceLog(FeaturesForRentInference):
    inferred_monthly_rent: float

    class Config:
        orm_mode = True


def _prepare_for_inference(features: List[FeaturesForRentInference]) -> np.ndarray:
    features_df = pd.DataFrame(jsonable_encoder(features))
    encoded, *_ = encode_data(features_df, encoder, scaler)
    return encoded, features_df


@app.post("/predict/")
def make_inference(
    features: List[FeaturesForRentInference],
    db_connection: Connection = Depends(get_connection),
) -> List[float]:
    prepared_features, features_df = _prepare_for_inference(features)
    result = model.predict(prepared_features)
    log_inference(db_connection, features_df, result)
    return result.tolist()


@app.get("/inferences/")
def get_last_inferences(
    limit: float = 10,
    session=Depends(get_session),
    response_model=List[InferenceLog],
) -> List[src.database.models.InferenceLog]:
    return get_inferences(session, limit)