## Dataset

The dataset used in this project is not included in this repository.

### Download

Download the dataset from the following source:

**Dataset:** [Customer churn](https://simplonline-v3-prod.s3.eu-west-3.amazonaws.com/media/file/csv/wa-fn-usec-telco-customer-churn-6ab92f83c7883379591536.csv)

After downloading, place the dataset in:

```text
data/raw/
```

The expected structure is:

```text
data/
└── raw/
    └── dataset.csv
```

## Process the data

From the repository root, run:

```bash
python -m src.preprocessing
```

The preprocessing step reads `data/raw/telco-customer.csv`, creates the engineered
features, makes a stratified 80/20 train/test split, fits imputers/scalers/encoders
on the training data only, and writes the following files to `data/processed/`:

- `X_train.csv`, `X_test.csv`: transformed model features
- `y_train.csv`, `y_test.csv`: encoded churn targets (`0 = No`, `1 = Yes`)
- `train_indexes.csv`, `test_indexes.csv`: original row indices for reproducibility
- `preprocessor.joblib`: fitted transformer for applying the same preprocessing to new data

The same flow is demonstrated in [`notebooks/02_preprocessing.ipynb`](../notebooks/02_preprocessing.ipynb).
