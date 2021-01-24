import pandas as pd
import pytest

import src.models as models


@pytest.fixture
def training_data():
    df = pd.read_csv("data/final/rent.csv")
    return df.drop(["listing_url", "monthly_rent"], axis="columns")


def test_encode_data(training_data):
    encoded, _, _ = models.encode_data(training_data)
    assert encoded.shape[0] == training_data.shape[0]
    assert encoded.dtype == float
    assert abs(encoded.mean()) < 1
