
from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def log_bundle(bundle_path: str | Path, tracking_uri: str = "sqlite:///mlflow.db") -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is optional. Install it with `pip install mlflow` to use tracking."
        ) from exc

    bundle_path = Path(bundle_path)
    bundle = joblib.load(bundle_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("client-churn")
    with mlflow.start_run(run_name="optimized-random-forest"):
        mlflow.log_params(bundle["best_params"])
        mlflow.log_metrics(
            {
                key: value
                for key, value in bundle["final_metrics"].items()
                if isinstance(value, (int, float))
            }
        )
        # The bundle is an imbalanced-learn pipeline; pickle preserves its
        # sampler and is appropriate because the artifact is produced locally.
        mlflow.sklearn.log_model(
            bundle["model"], name="model", serialization_format="pickle"
        )
        for name in ("cluster_metrics.csv", "cluster_profiles.csv", "model_comparison.csv"):
            path = bundle_path.parent / name
            if path.exists():
                mlflow.log_artifact(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="data/processed/model_bundle.joblib")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    log_bundle(args.bundle, args.tracking_uri)
