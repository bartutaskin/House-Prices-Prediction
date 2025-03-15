import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
import optuna
import os
import joblib

print(os.getcwd())
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 1000)


# Load dataset
def load_dataset(path_url):
    train_df = pd.read_csv(path_url + "/train.csv")
    test_df = pd.read_csv(path_url + "/test.csv")

    test_ids = test_df["Id"]
    train_df = train_df.drop("Id", axis=1)
    test_df = test_df.drop("Id", axis=1)

    return train_df, test_df, test_ids


# Grab columns according to their types
def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtype == "O"]
    num_but_cat = [
        col for col in df.columns if df[col].dtype != "O" and df[col].nunique() < cat_th
    ]
    cat_but_car = [
        col for col in df.columns if df[col].dtype == "O" and df[col].nunique() > car_th
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [
        col for col in df.columns if df[col].dtype != "O" and col != "SalePrice"
    ]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Categorical columns: {cat_cols}")
    print(f"Numerical columns: {num_cols}")
    print(f"High cardinality columns: {cat_but_car}")
    return cat_cols, num_cols, cat_but_car


# Handle missing values
def impute_missing_values(df):
    # Define imputation strategies
    impute_dict = {
        "None": [
            "PoolQC",
            "MiscFeature",
            "Alley",
            "Fence",
            "MasVnrType",
            "FireplaceQu",
            "GarageType",
            "GarageQual",
            "GarageCond",
            "BsmtExposure",
            "BsmtFinType2",
            "BsmtQual",
            "BsmtCond",
            "BsmtFinType1",
        ],
        0: [
            "GarageFinish",
            "BsmtFullBath",
            "BsmtHalfBath",
            "BsmtUnfSF",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "TotalBsmtSF",
            "GarageCars",
            "GarageArea",
        ],
        "mode": [
            "Electrical",
            "MSZoning",
            "Utilities",
            "Functional",
            "Exterior2nd",
            "Exterior1st",
            "KitchenQual",
            "SaleType",
        ],
        "median": ["LotFrontage"],
    }

    # Apply these strategies by iterating dict items. (Ex: value = "None", columns = ["PoolQC", ..., "BsmtFinType1"])
    for value, columns in impute_dict.items():
        if value == "None":
            for col in columns:
                df[col] = df[col].fillna("Not have")
        elif value == 0:
            for col in columns:
                df[col] = df[col].fillna(0)
        elif value == "mode":
            for col in columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif value == "median":
            for col in columns:
                df[col] = df[col].fillna(df[col].median())

    df.loc[df["GarageType"] == "Not have", "GarageYrBlt"] = 0
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())

    df.loc[df["MasVnrType"] == "Not have", "MasVnrArea"] == 0
    df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())

    return df


# Feature engineering
def create_features(df):
    df["New_BuildingAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["New_GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df["New_TotalBath"] = (
        df["BsmtFullBath"]
        + df["BsmtHalfBath"] * 0.5
        + df["FullBath"]
        + df["HalfBath"] * 0.5
    )
    df["New_OverallRate"] = df["OverallQual"] * df["OverallCond"]
    df["New_TotalPorchArea"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )
    df["New_AgeCategory"] = pd.cut(
        df["New_BuildingAge"],
        bins=[0, 10, 30, 50, 100, 200],
        labels=["0-10", "11-30", "31-50", "51-100", "100+"],
    )

    return df


# Encode categorical columns
def one_hot_encoder(df, categorical_cols):
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# Encode ordinal columns
def encode_ordinal_cols(df, mapping_dict):
    for col, mapping_values in mapping_dict.items():
        df[col] = df[col].map(mapping_values)
    return df


mapping_dict = {
    "PoolQC": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Not have": 0},
    "Fence": {"GdPrv": 4, "GdWo": 3, "MnPrv": 2, "MnWw": 1, "Not have": 0},
    "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not have": 0},
    "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not have": 0},
    "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not have": 0},
    "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},
    "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not have": 0},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "Not have": 0},
    "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not have": 0},
}


def train_xgb(X, y):
    xgb_model = XGBRegressor()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        xgb_model, X, y, cv=kf, scoring="neg_root_mean_squared_error"
    )
    return abs(scores.mean())


def tune_xgb_with_optuna(X, y):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        model = XGBRegressor(**params)
        score = cross_val_score(
            model, X, y, cv=5, scoring="neg_root_mean_squared_error"
        )
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    return study.best_params


def preprocess_data(train_df, test_df, mapping_dict):
    # Impute missing values
    train_df = impute_missing_values(train_df)
    test_df = impute_missing_values(test_df)

    # Feature engineering
    train_df = create_features(train_df)
    test_df = create_features(test_df)

    # Encode Ordinal Columns
    train_df = encode_ordinal_cols(train_df, mapping_dict)
    test_df = encode_ordinal_cols(test_df, mapping_dict)

    one_hot_cols = [
        col
        for col in train_df.columns
        if train_df[col].dtype == "O" and col not in list(mapping_dict.keys())
    ]
    one_hot_cols.extend(["MSSubClass", "New_AgeCategory"])

    # One-hot Encoding
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined_df = one_hot_encoder(combined_df, one_hot_cols)

    # Split back into train and test
    train_df = combined_df.iloc[: len(train_df)]
    test_df = combined_df.iloc[len(train_df) :]

    return train_df, test_df
    preprocessed_data = pd.DataFrame([features])

    preprocessed_data = impute_missing_values(preprocessed_data)

    preprocessed_data = create_features(preprocessed_data)

    preprocessed_data = encode_ordinal_cols(preprocessed_data, mapping_dict)
    print(preprocessed_data)
    one_hot_cols = [
        "MSZoning",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "Foundation",
        "Heating",
        "CentralAir",
        "Electrical",
        "Functional",
        "GarageType",
        "MiscFeature",
        "SaleType",
        "SaleCondition",
        "MSSubClass",
        "New_AgeCategory",
    ]

    preprocessed_data = one_hot_encoder(preprocessed_data, one_hot_cols)

    predicted_price = model.predict(preprocessed_data)

    return predicted_price


if __name__ == "__main__":
    # Load Data
    train_df, test_df, test_ids = load_dataset("c:/House-Prices-Prediction/data")

    # Preprocess Data
    train_df, test_df = preprocess_data(train_df, test_df, mapping_dict)

    # Split Data
    X = train_df.drop("SalePrice", axis=1)
    y = train_df["SalePrice"]

    # Train Model
    baseline_rmse = train_xgb(X, y)
    print("Baseline RMSE:", baseline_rmse)

    # Hyperparameter Tuning
    best_params = tune_xgb_with_optuna(X, y)
    print("Best Parameters:", best_params)

    # Train Final Model
    final_model = XGBRegressor(**best_params)
    final_model.fit(X, y)

    # Ensure the 'model' directory exists
    if not os.path.exists("model"):
        os.makedirs("model")

    joblib.dump(final_model, "model/xgb_model.pkl")
    print("Model saved successfully!")

    # Make Predictions
    X_test = test_df.drop("SalePrice", axis=1, errors="ignore")
    y_pred = final_model.predict(X_test)

    # Save Predictions
    pd.DataFrame({"Id": test_ids, "SalePrice": y_pred}).to_csv(
        "data/predictions2.csv", index=False
    )
