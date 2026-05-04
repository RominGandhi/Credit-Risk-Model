import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class CreditDataPreprocessor:
    """Preprocesses the merged credit card dataset for modeling.

    Pipeline steps:
        1. Handle missing values (median for numerical, mode for categorical)
        2. Handle outliers via IQR-based winsorization
        3. Encode categorical variables (label for binary, one-hot for multi-class)
        4. Scale numerical features with StandardScaler
        5. Split into stratified train/test sets and save to disk
    """

    def __init__(
        self,
        input_path="src/data/processed/merged_data.csv",
        output_dir="src/data/processed",
        target_col="default",
        test_size=0.2,
        random_state=42,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_cols = []
        self.categorical_cols = []
        self.drop_cols = ["ID"]

    def load_data(self):
        """Load the merged dataset and identify column types."""
        self.df = pd.read_csv(self.input_path)
        print(f"Loaded data: {self.df.shape}")

        self.categorical_cols = [
            col for col in self.df.select_dtypes(include=["object", "string"]).columns
            if col not in self.drop_cols
        ]
        self.numerical_cols = [
            col for col in self.df.select_dtypes(include=["number"]).columns
            if col not in self.drop_cols + [self.target_col]
        ]

        print(f"Numerical columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        return self.df

    def handle_missing_values(self):
        """Fill missing values: median for numerical, mode for categorical."""
        print("\n--- Missing Value Handling ---")
        missing_before = self.df.isnull().sum()
        missing_before = missing_before[missing_before > 0]
        print(f"Missing values BEFORE:\n{missing_before}\n")

        for col in self.numerical_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())

        for col in self.categorical_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        missing_after = self.df.isnull().sum()
        missing_after = missing_after[missing_after > 0]
        if missing_after.empty:
            print("Missing values AFTER: None")
        else:
            print(f"Missing values AFTER:\n{missing_after}")

        print(f"Shape after missing value handling: {self.df.shape}")
        return self.df

    def handle_outliers(self):
        """Detect and cap outliers using IQR-based winsorization on numerical columns."""
        print("\n--- Outlier Handling (IQR Winsorization) ---")

        for col in self.numerical_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if outlier_count > 0:
                print(f"  {col}: {outlier_count} outliers capped to [{lower:.2f}, {upper:.2f}]")
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)

        print(f"Shape after outlier handling: {self.df.shape}")
        return self.df

    def encode_categorical(self):
        """Label-encode binary categories; one-hot encode multi-class categories."""
        print("\n--- Categorical Encoding ---")

        binary_cols = [col for col in self.categorical_cols if self.df[col].nunique() == 2]
        multi_cols = [col for col in self.categorical_cols if self.df[col].nunique() > 2]

        # Label encoding for binary columns
        for col in binary_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"  Label encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # One-hot encoding for multi-class columns
        if multi_cols:
            self.df = pd.get_dummies(self.df, columns=multi_cols, drop_first=True, dtype=int)
            print(f"  One-hot encoded: {multi_cols}")

        print(f"Shape after encoding: {self.df.shape}")
        return self.df

    def scale_features(self):
        """Standardize numerical features using StandardScaler."""
        print("\n--- Feature Scaling ---")
        self.df[self.numerical_cols] = self.scaler.fit_transform(self.df[self.numerical_cols])
        print(f"Scaled {len(self.numerical_cols)} numerical columns")
        print(f"Shape after scaling: {self.df.shape}")
        return self.df

    def split_and_save(self):
        """Split into train/test sets with stratification and save as CSV."""
        print("\n--- Train/Test Split ---")

        # Drop ID column before splitting
        self.df.drop(columns=[c for c in self.drop_cols if c in self.df.columns], inplace=True)

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        print(f"Train set: {train_df.shape}")
        print(f"Test set:  {test_df.shape}")
        print(f"Train target distribution:\n{y_train.value_counts(normalize=True).round(4)}")
        print(f"Test target distribution:\n{y_test.value_counts(normalize=True).round(4)}")

        os.makedirs(self.output_dir, exist_ok=True)
        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"\nSaved train set to {train_path}")
        print(f"Saved test set to {test_path}")

        return train_df, test_df

    def run(self):
        """Execute the full preprocessing pipeline."""
        print("=" * 60)
        print("CREDIT DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        self.load_data()
        self.handle_missing_values()
        self.handle_outliers()
        self.encode_categorical()
        self.scale_features()
        train_df, test_df = self.split_and_save()

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        return train_df, test_df


if __name__ == "__main__":
    preprocessor = CreditDataPreprocessor()
    preprocessor.run()
