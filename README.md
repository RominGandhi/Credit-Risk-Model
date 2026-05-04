
# Credit Risk Model

Predicts whether a credit card applicant will default based on their application details and credit history. Three models are trained and compared: Random Forest, Logistic Regression, and XGBoost.

**Python Version: 3.13.1**

---

## Dataset

Two raw CSV files are required in `src/data/raw/`:

- `application_record.csv`: applicant demographic and financial information
- `credit_record.csv`: monthly credit status history per applicant

---

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate        # Mac/Linux
    .venv\Scripts\activate           # Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Running the Model from start to end

To run the full model:

```bash
python -m src.main
```

This will:
1. Merge the raw CSVs and create the target variable (`default`)
2. Engineer features (age, employment years, correlation/variance filtering)
3. Preprocess the data (missing values, outliers, train/test split, etc)
4. Train a Random Forest model
5. Train a Logistic Regression model
6. Train an XGBoost model
7. Compare all three models and print the best one by ROC AUC

Saved models are written to `src/models/saved/` and plots to `reports/figures/`.

---

## Running the Notebook

Open `notebooks/credit_defaulters.ipynb` and run all cells. The notebook will go through each step with outputs and visualizations. The visualization are also saved under `reports/figures/`.

---

## Project Organization

    ├── README.md
    ├── requirements.txt
    ├── LICENSE
    |
    ├── notebooks
    │   └── credit_defaulters.ipynb
    │
    ├── reports
    │   ├── figures                  <- Generated plots (confusion matrices, ROC curves, etc.)
    │   ├── README.md                <- YouTube video link
    │   ├── final_project_report     <- Final report (.pdf) and supporting files
    │   └── presentation             <- Final PowerPoint presentation
    │
    └── src
        │
        ├── main.py                  <- Runs the full model 
        │
        ├── data
        │   ├── raw                  <- Original CSV files
        │   ├── processed            <- Merged, feature-engineered, and split data
        │   └── make_dataset.py      <- Loads and merges raw data and creates target variable ('Default')
        │
        ├── preprocessing_data
        │   └── preprocessing.py     <- Missing values, outliers, train/test split, etc
        │
        ├── feature_engineering
        │   └── build_features.py    <- New features build from old ones and removing low-variance and high-correlation features
        │
        ├── models
        │   ├── train_model.py       <- Random Forest, Logistic Regression, XGBoost model training
        │   └── predict_model.py     <- Loads saved models and compares performance
        │
        └── visualization
            └── visualize.py         <- EDA plots (histograms, bar charts, heatmap, boxplots)
