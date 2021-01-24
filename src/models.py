import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = [
    "city",
    "district",
    "street",
    "building_type",
    "heating_type",
    "equipment",
]


def encode_data(
    rent_features: pd.DataFrame,
    one_hot_encoder: OneHotEncoder = None,
    scaler: StandardScaler = None,
) -> Tuple[np.ndarray, OneHotEncoder, StandardScaler]:
    """Encodes categorical features using one-hot encoding.

    If passed an existing encoder, will use that one.
    """
    if not one_hot_encoder:
        one_hot_encoder = OneHotEncoder().fit(rent_features[CATEGORICAL_COLUMNS])
    encoded_data = one_hot_encoder.transform(
        rent_features[CATEGORICAL_COLUMNS]
    ).todense()

    if not scaler:
        scaler = StandardScaler().fit(
            rent_features.drop(CATEGORICAL_COLUMNS, axis="columns")
        )
    scaled_data = scaler.transform(
        rent_features.drop(CATEGORICAL_COLUMNS, axis="columns")
    )

    encoded_data = np.concatenate(
        [encoded_data, scaled_data],
        axis=-1,
    )

    return encoded_data, one_hot_encoder, scaler


def describe_model(
    model: Lasso,
    encoder: OneHotEncoder,
    feature_columns,
    test_features,
    test_target,
):
    """Prints information that helps to interpret the model."""
    print(f"Score: {model.score(test_features, test_target)}")
    pred = model.predict(test_features)
    mae = metrics.mean_absolute_error(test_target, pred)
    mape = metrics.mean_absolute_percentage_error(test_target, pred)
    print(f"Mean absolute error: {mae}")
    print(f"Mean absolute percentage error: {mape}")


def main(
    rent_dataframe: pd.DataFrame,
) -> Tuple[Lasso, OneHotEncoder, StandardScaler]:
    """Main entry point for this module.

    Encodes data with one hot encoder,
    Trains linear regression model,
    Prints model's score and weights,
    Returns trained model and used used encoder for further use.
    """
    encoded_features, encoder, scaler = encode_data(
        rent_dataframe.drop("monthly_rent", axis="columns")
    )
    train_features, test_features, train_rent, test_rent = train_test_split(
        encoded_features,
        rent_dataframe["monthly_rent"],
    )
    model = Lasso(fit_intercept=False)
    model.fit(train_features, train_rent)
    describe_model(
        model,
        encoder,
        rent_dataframe.drop("monthly_rent", axis="columns").columns,
        test_features,
        test_rent,
    )
    return model, encoder, scaler


if __name__ == "__main__":
    data = pd.read_csv("data/final/rent.csv", index_col=0).drop(
        "listing_url", axis="columns"
    )
    model, encoder, scaler = main(data)
    with open("models/model.pck", "wb") as model_pck:
        pickle.dump(model, model_pck)
    with open("models/encoder.pck", "wb") as ohe_pck:
        pickle.dump(encoder, ohe_pck)
    with open("models/scaler.pck", "wb") as scaler_pck:
        pickle.dump(scaler, scaler_pck)
