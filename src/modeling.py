from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .preprocessing import TARGET, create_preprocessor, engineer_features


def make_models(y: pd.Series) -> dict[str, BaseEstimator]:
    negative, positive = np.bincount(y.astype(int))
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            scale_pos_weight=negative / positive,
            random_state=42,
            n_jobs=1,
        ),
    }


def make_pipeline(model: BaseEstimator, include_cluster: bool = False) -> ImbPipeline:
    return ImbPipeline(
        [
            ("preprocessor", create_preprocessor(include_cluster=include_cluster)),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ]
    )


def evaluate_model(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series
) -> dict[str, Any]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(
            y, predictions, output_dict=True, zero_division=0
        ),
    }


def fit_clusters(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cluster_preprocessor = create_preprocessor()
    train_matrix = cluster_preprocessor.fit_transform(X_train)
    test_matrix = cluster_preprocessor.transform(X_test)
    pca = PCA(n_components=0.95, random_state=random_state)
    train_pca = pca.fit_transform(train_matrix)
    test_pca = pca.transform(test_matrix)

    candidates: list[dict[str, Any]] = []
    for k in range(2, 9):
        estimator = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = estimator.fit_predict(train_pca)
        candidates.append(
            {
                "algorithm": "kmeans",
                "n_clusters": k,
                "inertia": float(estimator.inertia_),
                "silhouette": float(silhouette_score(train_pca, labels)),
                "davies_bouldin": float(davies_bouldin_score(train_pca, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(train_pca, labels)),
            }
        )
    metrics = pd.DataFrame(candidates)
    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "n_clusters"])
    kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=random_state)
    train_labels = kmeans.fit_predict(train_pca)
    test_labels = kmeans.predict(test_pca)

    # Alternatives are evaluated for comparison; K-Means remains the deployable model
    # because DBSCAN has no reliable transform for unseen customers.
    alternatives = []
    for name, estimator in (
        ("dbscan", DBSCAN(eps=2.5, min_samples=5)),
        (
            "agglomerative",
            AgglomerativeClustering(n_clusters=best_k, linkage="ward"),
        ),
    ):
        labels = estimator.fit_predict(train_pca)
        valid = labels != -1
        if valid.sum() > 1 and len(np.unique(labels[valid])) > 1:
            alternatives.append(
                {
                    "algorithm": name,
                    "silhouette": float(silhouette_score(train_pca[valid], labels[valid])),
                    "davies_bouldin": float(
                        davies_bouldin_score(train_pca[valid], labels[valid])
                    ),
                    "calinski_harabasz": float(
                        calinski_harabasz_score(train_pca[valid], labels[valid])
                    ),
                    "noise_points": int((labels == -1).sum()),
                }
            )
    artifacts = {
        "preprocessor": cluster_preprocessor,
        "pca": pca,
        "kmeans": kmeans,
        "metrics": pd.concat([metrics, pd.DataFrame(alternatives)], ignore_index=True),
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_pca": train_pca,
        "test_pca": test_pca,
    }
    return artifacts, artifacts["metrics"]


def train(
    input_path: str | Path,
    output_dir: str | Path = "data/processed",
    random_state: int = 42,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = engineer_features(pd.read_csv(input_path))
    X = data.drop(columns=TARGET)
    y = data[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn contains values other than 'Yes' and 'No'")
    y = y.astype(int)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    cluster_artifacts, cluster_metrics = fit_clusters(X_train, X_test, random_state)
    X_train_cluster = X_train.copy()
    X_test_cluster = X_test.copy()
    X_train_cluster["cluster"] = cluster_artifacts["train_labels"].astype(str)
    X_test_cluster["cluster"] = cluster_artifacts["test_labels"].astype(str)

    baseline_results: list[dict[str, Any]] = []
    fitted_with_cluster: dict[str, Any] = {}
    for name, estimator in make_models(y_train).items():
        baseline = make_pipeline(estimator, include_cluster=False)
        baseline.fit(X_train, y_train)
        baseline_results.append(
            {"model": name, "features": "original", **evaluate_model(baseline, X_test, y_test)}
        )
        clustered = make_pipeline(estimator, include_cluster=True)
        clustered.fit(X_train_cluster, y_train)
        fitted_with_cluster[name] = clustered
        baseline_results.append(
            {
                "model": name,
                "features": "original + cluster",
                **evaluate_model(clustered, X_test_cluster, y_test),
            }
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    optimization_base = make_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
        include_cluster=True,
    )
    search = RandomizedSearchCV(
        optimization_base,
        {
            "model__n_estimators": [150, 250, 350],
            "model__max_depth": [None, 8, 15, 25],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        n_iter=8,
        scoring="f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train_cluster, y_train)
    final_metrics = evaluate_model(search.best_estimator_, X_test_cluster, y_test)
    baseline_frame = pd.DataFrame(baseline_results)
    baseline_frame.drop(columns=["confusion_matrix", "classification_report"]).to_csv(
        output / "model_comparison.csv", index=False
    )
    cluster_metrics.to_csv(output / "cluster_metrics.csv", index=False)
    pd.DataFrame(
        {"cluster": np.r_[cluster_artifacts["train_labels"], cluster_artifacts["test_labels"]]}
    ).to_csv(output / "cluster_labels.csv", index=False)
    profiles = _cluster_profiles(
        pd.concat([X_train, X_test], ignore_index=True),
        np.r_[cluster_artifacts["train_labels"], cluster_artifacts["test_labels"]],
    )
    profiles.to_csv(output / "cluster_profiles.csv", index=False)
    bundle = {
        "model": search.best_estimator_,
        "cluster_preprocessor": cluster_artifacts["preprocessor"],
        "cluster_pca": cluster_artifacts["pca"],
        "kmeans": cluster_artifacts["kmeans"],
        "cluster_metrics": cluster_metrics,
        "best_params": search.best_params_,
        "final_metrics": final_metrics,
        "feature_columns": list(X.columns),
    }
    joblib.dump(bundle, output / "model_bundle.joblib")
    return {"baseline_results": baseline_frame, "cluster_metrics": cluster_metrics, **bundle}


def _cluster_profiles(data: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    profile = data.assign(cluster=labels).groupby("cluster").agg(
        customers=("cluster", "size"),
        tenure=("tenure", "mean"),
        monthly_charges=("MonthlyCharges", "mean"),
        total_charges=("TotalCharges", "mean"),
        contract_level=("ContractLevel", "mean"),
        number_of_services=("NumServices", "mean"),
    )
    return profile.reset_index().round(2)


def predict_customer(bundle_path: str | Path, customer: dict[str, Any]) -> dict[str, Any]:
    bundle = joblib.load(bundle_path)
    frame = pd.DataFrame([customer])
    engineered = engineer_features(frame)
    cluster_matrix = bundle["cluster_preprocessor"].transform(engineered)
    cluster_point = bundle["cluster_pca"].transform(cluster_matrix)
    cluster = int(bundle["kmeans"].predict(cluster_point)[0])
    engineered["cluster"] = str(cluster)
    probability = float(bundle["model"].predict_proba(engineered)[:, 1][0])
    return {
        "cluster": cluster,
        "churn": int(probability >= 0.5),
        "churn_probability": probability,
    }
