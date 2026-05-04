from src.data.make_dataset import CreditDatasetBuilder
from src.preprocessing_data.preprocessing import CreditDataPreprocessor
from src.feature_engineering.build_features import FeatureEngineer
from src.models.train_model import RandomForestModel, LogisticRegressionModel, XGBoostModel
from src.models.predict_model import ModelComparator


def main():
    """Run the full credit default prediction model from start to end.

    Steps:
        1. Build dataset from raw CSVs (application + credit records)
        2. Engineer features (derived columns, low-variance/correlation removal)
        3. Preprocess (missing values, outliers, train/test split, etc)
        4. Train Random Forest model
        5. Train Logistic Regression model
        6. Train XGBoost model
        7. Compare all models and select best
    """

    print("=" * 60)
    print("CREDIT DEFAULT PREDICTION MODEL")
    print("=" * 60)

    # Step 1: Load and merge raw data into merged_data.csv
    print("\n[1/7] Building dataset...")
    builder = CreditDatasetBuilder()
    builder.build()

    # Step 2: Feature engineering on merged data, saves to features.csv
    print("\n[2/7] Engineering features...")
    engineer = FeatureEngineer()
    engineer.run()

    # Step 3: Preprocess the feature-engineered data, split into train/test
    print("\n[3/7] Preprocessing data...")
    preprocessor = CreditDataPreprocessor(
        input_path="src/data/processed/features.csv"
    )
    preprocessor.run()

    # Step 4: Train Random Forest
    print("\n[4/7] Training Random Forest...")
    rf = RandomForestModel()
    rf_metrics = rf.run()

    # Step 5: Train Logistic Regression
    print("\n[5/7] Training Logistic Regression...")
    lr = LogisticRegressionModel()
    lr_metrics = lr.run()

    # Step 6: Train XGBoost
    print("\n[6/7] Training XGBoost...")
    xgb = XGBoostModel()
    xgb_metrics = xgb.run()

    # Step 7: Compare all models and select the best one
    print("\n[7/7] Comparing models...")
    comparator = ModelComparator()
    comparison_df, best_model = comparator.run()

    # Final summary
    print("\n" + "=" * 60)
    print("MODEL COMPLETE")
    print("=" * 60)
    print(f"\nBest model: {best_model}")
    print(f"\nAll metrics:")
    print(comparison_df.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()