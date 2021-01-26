import os

import requests
import numpy as np
import pytest


@pytest.fixture
def features_for_inference():
    return [
        {
            "city": "kaune",
            "district": "senamiestyje",
            "latitude": 54.895115,
            "longitude": 23.885151,
            "street": "t-daugirdo-g",
            "floor_area_m2": 65.0,
            "number_of_rooms": 3.0,
            "floor": 2.0,
            "number_of_floors": 3.0,
            "build_year": 1987.0,
            "building_type": "Mūrinis",
            "heating_type": "Centrinis",
            "equipment": "Įrengtas",
        },
        {
            "city": "vilniuje",
            "district": "naujamiestyje",
            "latitude": 54.674321,
            "longitude": 25.290807,
            "street": "m-dauksos-g",
            "floor_area_m2": 65.0,
            "number_of_rooms": 3.0,
            "floor": 1.0,
            "number_of_floors": 3.0,
            "build_year": 1940.0,
            "building_type": "Mūrinis",
            "heating_type": "Geoterminis",
            "equipment": "Įrengtas",
        },
        {
            "city": "vilniuje",
            "district": "grigiskese",
            "latitude": 54.668728,
            "longitude": 25.099926,
            "street": "kovo-11-osios-g",
            "floor_area_m2": 75.0,
            "number_of_rooms": 3.0,
            "floor": 5.0,
            "number_of_floors": 6.0,
            "build_year": 2002.0,
            "building_type": "Mūrinis",
            "heating_type": "Centrinis",
            "equipment": "Įrengtas",
        },
        {
            "city": "vilniuje",
            "district": "naujamiestyje",
            "latitude": 54.672993,
            "longitude": 25.278603,
            "street": "kauno-g",
            "floor_area_m2": 66.16,
            "number_of_rooms": 2.0,
            "floor": 5.0,
            "number_of_floors": 6.0,
            "build_year": 1980.0,
            "building_type": "Blokinis",
            "heating_type": "Centrinis",
            "equipment": "Įrengtas",
        },
        {
            "city": "kaune",
            "district": "centre",
            "latitude": 54.889328,
            "longitude": 23.936227,
            "street": "tunelio-g",
            "floor_area_m2": 25.0,
            "number_of_rooms": 1.0,
            "floor": 1.0,
            "number_of_floors": 2.0,
            "build_year": 1939.0,
            "building_type": "Medinis",
            "heating_type": "Dujinis",
            "equipment": "Įrengtas",
        },
    ]


@pytest.fixture
def targets_for_inference():
    return np.array([950.0, 600.0, 390.0, 690.0, 280.0])


def test_predict(features_for_inference, targets_for_inference):
    """Right now this writes data to production database.

    I'd have to set up a test database if this was a real job.
    """
    url = f"http://{os.environ['SERVER']}:{os.environ['PORT']}/predict/"
    response = requests.post(url, json=features_for_inference)
    inferred = np.array(response.json())
    for idx, (true, pred) in enumerate(zip(targets_for_inference, inferred)):
        assert pytest.approx(true, 300) == pred


def test_get_inferences():
    """Relies on prod database having data"""
    url = f"http://{os.environ['SERVER']}:{os.environ['PORT']}/inferences/"
    response = requests.get(url)
    logs = response.json()
    assert len(logs) == 10