from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Churn"
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlyCharges",
    "HasInternetService",
    "NumServices",
    "ChargesPerService",
    "ContractLevel",
]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "PaymentMethod",
]


def create_preprocessor() -> ColumnTransformer:
 
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:

    required_columns = {"customerID", TARGET, *NUMERIC_FEATURES[:4], *CATEGORICAL_FEATURES}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    result = data.copy()
    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    result["AvgMonthlyCharges"] = (
        result["TotalCharges"] / result["tenure"].replace(0, np.nan)
    ).fillna(result["MonthlyCharges"])
    result["HasInternetService"] = result["InternetService"].ne("No").astype(int)
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    result["NumServices"] = result[service_columns].eq("Yes").sum(axis=1)
    result["ChargesPerService"] = np.where(
        result["NumServices"].eq(0),
        0,
        result["MonthlyCharges"] / result["NumServices"],
    )
    result["ContractLevel"] = result["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    if result["ContractLevel"].isna().any():
        raise ValueError("Contract contains an unknown category")
    return result.drop(columns=["customerID", "Contract"])


def process_data(
    input_path: str | Path,
    output_dir: str | Path = "data/processed",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = engineer_features(pd.read_csv(input_path))
    X = data.drop(columns=[TARGET])
    y = data[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn contains values other than 'Yes' and 'No'")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.astype(int),
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    preprocessor = create_preprocessor()
    X_train_processed = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=preprocessor.get_feature_names_out(),
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out(),
        index=X_test.index,
    )

    X_train_processed.to_csv(output_path / "X_train.csv", index=False)
    X_test_processed.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False, header=[TARGET])
    y_test.to_csv(output_path / "y_test.csv", index=False, header=[TARGET])
    pd.Series(X_train.index, name="index").to_csv(
        output_path / "train_indexes.csv", index=False
    )
    pd.Series(X_test.index, name="index").to_csv(
        output_path / "test_indexes.csv", index=False
    )
    joblib.dump(preprocessor, output_path / "preprocessor.joblib")
    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    process_data("data/raw/telco-customer.csv")