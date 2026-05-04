import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FeatureEngineer:
    """Creates derived features and filters out low-value columns.

    Pipeline steps:
        1. Create derived features (age, employment years, income ratios)
        2. Label-encode remaining categorical columns
        3. Remove low-variance features (threshold < 0.01)
        4. Remove highly correlated features (threshold > 0.9)
        5. Rank features by mutual information with the target
        6. Save the engineered dataset to CSV
    """

    def __init__(self, csv_path="src/data/processed/merged_data.csv", out_dir="src/data/processed"):
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.df = None

    def load(self):
        """Load the merged dataset from CSV."""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded data: {self.df.shape}")
        return self.df

    def create_features(self):
        """Create derived features from raw columns.

        - age_years: absolute age derived from DAYS_BIRTH
        - years_employed: employment duration from DAYS_EMPLOYED
        - income_per_member: income divided by family size
        - income_per_child: income divided by number of children
        """
        if "DAYS_BIRTH" in self.df.columns:
            self.df["age_years"] = abs(self.df["DAYS_BIRTH"]) // 365
            self.df = self.df.drop(columns=["DAYS_BIRTH"])

        if "DAYS_EMPLOYED" in self.df.columns:
            # 365243 is a placeholder in the raw data for unemployed applicants
            self.df["DAYS_EMPLOYED"] = self.df["DAYS_EMPLOYED"].replace(365243, 0)
            self.df["years_employed"] = abs(self.df["DAYS_EMPLOYED"]) // 365
            self.df = self.df.drop(columns=["DAYS_EMPLOYED"])

        if "AMT_INCOME_TOTAL" in self.df.columns and "CNT_FAM_MEMBERS" in self.df.columns:
            self.df["income_per_member"] = self.df["AMT_INCOME_TOTAL"] / self.df["CNT_FAM_MEMBERS"].replace(0, 1)

        if "AMT_INCOME_TOTAL" in self.df.columns and "CNT_CHILDREN" in self.df.columns:
            self.df["income_per_child"] = self.df.apply(
                lambda row: row["AMT_INCOME_TOTAL"] / row["CNT_CHILDREN"] if row["CNT_CHILDREN"] > 0 else 0, axis=1
            )

        print(f"New features created. Shape: {self.df.shape}")
        return self.df

    def encode_categoricals(self):
        """Label-encode all remaining object-type columns."""
        cat_cols = self.df.select_dtypes(include="object").columns
        le = LabelEncoder()

        for col in cat_cols:
            self.df[col] = self.df[col].fillna("Unknown")
            self.df[col] = le.fit_transform(self.df[col].astype(str))

        print(f"Encoded {len(cat_cols)} categorical columns")
        return self.df

    def remove_low_variance(self, threshold=0.01):
        """Drop feature columns with variance below the threshold."""
        drop_cols = []
        for col in self.df.columns:
            if col == "default":
                continue
            if self.df[col].var() < threshold:
                drop_cols.append(col)

        self.df = self.df.drop(columns=drop_cols)
        print(f"Removed low variance columns: {drop_cols}")
        return self.df

    def remove_high_correlation(self, threshold=0.9):
        """Drop one of each pair of features correlated above the threshold."""
        features = self.df.drop(columns=["ID", "default"], errors="ignore")
        corr_matrix = features.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]

        self.df = self.df.drop(columns=drop_cols)
        print(f"Removed highly correlated columns: {drop_cols}")
        return self.df

    def correlation_with_target(self):
        """Print each feature's absolute correlation with the target column."""
        drop = ["ID", "default"]
        X = self.df.drop(columns=drop, errors="ignore")
        corr = X.corrwith(self.df["default"]).abs().sort_values(ascending=False)

        print("\nFeature correlation with target:")
        print(corr.to_string())
        return corr

    def rank_features(self, k=15):
        """Rank the top-k features by mutual information with the target."""
        X = self.df.drop(columns=["ID", "default"], errors="ignore")
        y = self.df["default"]

        selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)

        scores = pd.DataFrame({
            "feature": X.columns,
            "score": selector.scores_,
        }).sort_values("score", ascending=False)

        print("\nFeature Ranking (Mutual Information):")
        print(scores.to_string(index=False))
        return scores

    def save(self):
        """Save the engineered dataset to features.csv."""
        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, "features.csv")
        self.df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

    def run(self):
        """Execute the full feature engineering pipeline."""
        print("=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)

        self.load()
        self.create_features()
        self.encode_categoricals()
        self.remove_low_variance()
        self.remove_high_correlation()
        self.correlation_with_target()
        self.rank_features()
        self.save()

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.run()
