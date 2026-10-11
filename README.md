# Client churn prediction

Cette version fournit un pipeline reproductible de segmentation et de prédiction.

## Entraîner le modèle

Après avoir placé le fichier dans `data/raw/telco-customer.csv` :

```bash
python -m src.train
```

Le script ajuste le prétraitement uniquement sur l'entraînement, compare les cinq
classifieurs demandés, évalue K-Means/DBSCAN/Agglomerative, optimise une forêt
aléatoire par validation croisée et enregistre le modèle dans
`data/processed/model_bundle.joblib`. Le cluster est ajusté sur l'entraînement et
ajouté ensuite comme variable de classification, sans jamais utiliser `Churn` pour
la segmentation.

## Suivi MLflow

MLflow est optionnel pour l'entraînement local. Une fois installé :

```bash
python -m src.mlflow_tracking --bundle data/processed/model_bundle.joblib
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Application

```bash
streamlit run app/streamlit_app.py
```

L'application affiche le cluster, la probabilité de churn et la décision associée.
Elle peut aussi être lancée avec `docker compose up --build`.