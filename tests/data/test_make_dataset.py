import pandas as pd
import pytest

import src.data.make_dataset as make_dataset


@pytest.fixture
def raw_rent_data():
    return make_dataset.load_data("data/raw/rent.csv").head(5)


def test_load_data(raw_rent_data):
    assert (
        raw_rent_data.columns
        == (
            [
                "city",
                "distances",
                "district",
                "latitude",
                "listing_url",
                "longitude",
                "number_of_crimes_within_500_meters",
                "object_details",
                "street",
            ]
        )
    ).all()


def test_expand_dataset(raw_rent_data):
    df = make_dataset.expand_data(raw_rent_data)
    assert set(df.columns) == set(
        [
            "city",
            "distances",
            "district",
            "latitude",
            "listing_url",
            "longitude",
            "number_of_crimes_within_500_meters",
            "object_details",
            "street",
            "Plotas:",
            "Kaina mėn.:",
            "Kambarių sk.:",
            "Aukštas:",
            "Aukštų sk.:",
            "Metai:",
            "Pastato tipas:",
            "Šildymas:",
            "Įrengimas:",
            "Ypatybės:",
            "Papildomos patalpos:",
            "Papildoma įranga:",
            "Apsauga:",
            "Buto numeris:",
            "Namo numeris:",
        ]
    )


def test_make_intermediate():
    df = make_dataset.make_intermediate("data/raw/rent.csv")
    assert set(df.columns) == set(
        [
            "city",
            "district",
            "latitude",
            "listing_url",
            "longitude",
            "street",
            "floor_area_m2",
            "monthly_rent",
            "number_of_rooms",
            "floor",
            "number_of_floors",
            "build_year",
            "building_type",
            "heating_type",
            "equipment",
        ]
    )
    assert not df.isna().any(axis=None)
    assert df.shape[0] > 100
    numeric_cols = [
        "latitude",
        "longitude",
        "floor_area_m2",
        "monthly_rent",
        "number_of_rooms",
        "floor",
        "number_of_floors",
        "build_year",
    ]
    for col in numeric_cols:
        assert df[col].dtype == float
