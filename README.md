# Team-Adopted---ML

Machine-learning solution for the **CS Club Online News Popularity** challenge.
The project predicts the popularity class of an online news article and uses
**Macro F1** as its primary evaluation metric.

## Project structure

```text
.
├── solution.ipynb       # End-to-end training and submission notebook
├── data/                # Local challenge data; ignored by Git
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── README.md
```

The notebook expects the following files under `data/`:

- `train.csv`: labeled training data with `target_popularity`
- `test.csv`: unlabeled data used for prediction
- `sample_submission.csv`: example of the required submission format

The training data contains 60 input features, an `id` column, and a
`target_popularity` class with values `A`, `B`, `C`, `D`, or `E`.

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- SciPy
- scikit-learn
- LightGBM
- XGBoost
- CatBoost

Install the main dependencies with:

```bash
python -m pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost jupyter
```

PyTorch is optional and is only used by the notebook when the optional
GPU-related path is enabled.

## Run the notebook

From the repository root, start Jupyter:

```bash
jupyter notebook solution.ipynb
```

Run the notebook cells from top to bottom. The workflow:

1. Loads and inspects the training and test data.
2. Applies feature engineering.
3. Uses adversarial validation to check for train/test distribution drift.
4. Compares feature-engineering and drift-removal choices with cross-validation.
5. Trains multi-seed booster models, including LightGBM, XGBoost, and CatBoost.
6. Combines predictions and creates a competition submission.

Some experiments are configured for GPU execution and may require a compatible
CUDA installation. CPU execution remains suitable for the standard analysis,
but may take longer.

## Submission format

A valid submission must contain exactly two columns:

```text
id,target_popularity
```

There must be one predicted row for every row in `data/test.csv`. Use
`data/sample_submission.csv` as the format reference.

## Reproducibility

- The notebook uses seeded cross-validation and multi-seed model training.
- Missing values represented as `NA` are handled during data loading.
- Keep the notebook's cells in order because later experiments depend on
  features and settings defined earlier.
- Generated submissions and local datasets are intentionally excluded from Git.

## Data and challenge terms

The challenge data is not included in version control. Obtain and use it
according to the original challenge's data-access and submission terms.
