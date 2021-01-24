import ast

import pandas as pd


def load_data(filename: str):
    """WARNING: evaluates some columns as python code.

    Make sure data is from trusted source.
    """
    return pd.read_csv(filename, converters={"object_details": ast.literal_eval})


def expand_data(df: pd.DataFrame):
    """Expands dicts in data to separate columns."""
    return pd.concat([df, pd.json_normalize(df.object_details)], axis="columns")


def drop_and_rename_columns(dataframe: pd.DataFrame):
    dataframe = dataframe.dropna(how="all", axis="columns").copy()
    dataframe.drop(
        [
            "Ypatybės:",
            "Papildomos patalpos:",
            "Papildoma įranga:",
            "Apsauga:",
            "Namo numeris:",
            "Buto numeris:",
            "Pastato energijos suvartojimo klasė:",
            "object_details",
            "distances",
        ],
        axis="columns",
        inplace=True,
    )
    dataframe.dropna(axis="rows", inplace=True)
    dataframe.rename(
        columns={
            "Plotas:": "floor_area_m2",
            "Kaina mėn.:": "monthly_rent",
            "Kambarių sk.:": "number_of_rooms",
            "Aukštas:": "floor",
            "Aukštų sk.:": "number_of_floors",
            "Metai:": "build_year",
            "Pastato tipas:": "building_type",
            "Šildymas:": "heating_type",
            "Įrengimas:": "equipment",
        },
        inplace=True,
    )
    return dataframe


def transform_to_numeric_values(dataframe: pd.DataFrame, inplace=False):
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
    if not inplace:
        dataframe = dataframe.copy()
    dataframe.loc[:, "floor_area_m2"] = dataframe.loc[:, "floor_area_m2"].str.replace(
        r"m²|\s", "", regex=True
    )
    dataframe.loc[:, "monthly_rent"] = dataframe.loc[:, "monthly_rent"].str.replace(
        r"€|\s", "", regex=True
    )
    dataframe.loc[:, "build_year"] = dataframe.loc[:, "build_year"].str.replace(
        r"^.*?(?P<build_year>\d{4}).*$", r"\g<build_year>", regex=True
    )
    for col in numeric_cols:
        try:
            dataframe.loc[:, col] = dataframe.loc[:, col].str.replace(",", ".")
        except AttributeError:
            pass
        dataframe.loc[:, col] = dataframe.loc[:, col].astype(float)
    return dataframe


def make_intermediate(raw_data_filename: str = "data/raw/rent.csv"):
    dataframe = load_data(raw_data_filename)
    dataframe = expand_data(dataframe)
    dataframe = drop_and_rename_columns(dataframe)
    dataframe = transform_to_numeric_values(dataframe)
    return dataframe


if __name__ == "__main__":
    df = make_intermediate()
    df.to_csv("data/intermediate/rent.csv")