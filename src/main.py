import pickle
from typing import List

from fastapi import Depends, FastAPI
from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.models import encode_data
from src.database.repository import log_inference
from src.database import Engine, get_connection
import src.database.models

app = FastAPI()
model = pickle.load(open("models/model.pck", "rb"))
encoder = pickle.load(open("models/encoder.pck", "rb"))
scaler = pickle.load(open("models/scaler.pck", "rb"))
engine = Engine()

src.database.models.Base.metadata.create_all(bind=engine)


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


def _prepare_for_inference(features: List[FeaturesForRentInference]) -> np.ndarray:
    features_df = pd.DataFrame(jsonable_encoder(features))
    encoded, *_ = encode_data(features_df, encoder, scaler)
    return encoded, features_df


@app.post("/predict/")
def make_inference(
    features: List[FeaturesForRentInference],
    db_session: Session = Depends(lambda: get_connection(engine)),
):
    prepared_features, features_df = _prepare_for_inference(features)
    result = model.predict(prepared_features)
    log_inference(db_session, features_df, result)
    return result.tolist()
