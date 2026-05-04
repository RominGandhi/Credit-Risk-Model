import os
import pandas as pd


class CreditDatasetBuilder:
    """Loads, merges, and processes application and credit record data
    into a single dataset suitable for credit default prediction."""

    def __init__(self, raw_dir="src/data/raw", processed_dir="src/data/processed"):
        """Initialize with paths to raw and processed data directories.

        Args:
            raw_dir: Path to the directory containing raw CSV files.
            processed_dir: Path to the directory for saving processed output.
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.application_df = None
        self.credit_df = None
        self.merged_df = None

    def load_data(self):
        """Load application_record.csv and credit_record.csv from the raw directory.

        Returns:
            Tuple of (application_df, credit_df) DataFrames.
        """
        app_path = os.path.join(self.raw_dir, "application_record.csv")
        credit_path = os.path.join(self.raw_dir, "credit_record.csv")

        self.application_df = pd.read_csv(app_path)
        self.credit_df = pd.read_csv(credit_path)

        print(f"Application records loaded: {self.application_df.shape}")
        print(f"Credit records loaded: {self.credit_df.shape}")

        return self.application_df, self.credit_df

    def create_target_variable(self):
        """Create a binary default target from the credit record STATUS column.

        An applicant is labeled as default=1 if they have a STATUS of
        2, 3, 4, or 5 (indicating 60+ days overdue) at any point in their
        credit history. Otherwise default=0.

        Returns:
            DataFrame with one row per ID and a 'default' column.
        """
        # STATUS values 2-5 indicate increasingly severe delinquency
        self.credit_df["is_overdue"] = self.credit_df["STATUS"].isin(
            ["2", "3", "4", "5", 2, 3, 4, 5]
        ).astype(int)

        target_df = (
            self.credit_df.groupby("ID")["is_overdue"]
            .max()
            .reset_index()
            .rename(columns={"is_overdue": "default"})
        )

        print(f"Target variable created for {len(target_df)} applicants")
        return target_df

    def aggregate_credit_records(self):
        """Aggregate monthly credit records to one row per applicant.

        Computes summary statistics from the credit history including
        total months on record and status distribution counts.

        Returns:
            DataFrame with aggregated credit features per applicant.
        """
        # Count occurrences of each STATUS value per applicant
        status_counts = (
            self.credit_df.groupby("ID")["STATUS"]
            .value_counts()
            .unstack(fill_value=0)
            .add_prefix("status_")
            .reset_index()
        )

        # Compute total months on record per applicant
        months_on_record = (
            self.credit_df.groupby("ID")["MONTHS_BALANCE"]
            .agg(months_count="count", months_min="min", months_max="max")
            .reset_index()
        )

        aggregated_df = pd.merge(status_counts, months_on_record, on="ID", how="outer")
        print(f"Credit records aggregated: {aggregated_df.shape}")
        return aggregated_df

    def merge_datasets(self):
        """Merge application records with aggregated credit data and target variable.

        Combines the application data, aggregated credit features, and default
        target into a single DataFrame with one row per applicant.

        Returns:
            Merged DataFrame ready for modeling.
        """
        target_df = self.create_target_variable()
        aggregated_credit_df = self.aggregate_credit_records()

        # Merge application data with aggregated credit features
        self.merged_df = pd.merge(
            self.application_df, aggregated_credit_df, on="ID", how="inner"
        )

        # Merge in the target variable
        self.merged_df = pd.merge(self.merged_df, target_df, on="ID", how="inner")

        print(f"Merged dataset shape: {self.merged_df.shape}")
        return self.merged_df

    def save_processed_data(self):
        """Save the merged dataset to the processed data directory as CSV.

        Creates the output directory if it does not exist.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        output_path = os.path.join(self.processed_dir, "merged_data.csv")
        self.merged_df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

    def print_summary(self):
        """Print a summary of the merged dataset including shape, columns,
        and target variable distribution."""
        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Shape: {self.merged_df.shape}")
        print(f"\nColumns ({len(self.merged_df.columns)}):")
        for col in self.merged_df.columns:
            print(f"  - {col}")
        print(f"\nTarget variable distribution:")
        dist = self.merged_df["default"].value_counts()
        total = len(self.merged_df)
        for label, count in dist.items():
            pct = count / total * 100
            print(f"  default={label}: {count} ({pct:.2f}%)")
        print("=" * 50)

    def build(self):
        """Run the full pipeline: load, merge, save, and summarize.

        Returns:
            The final merged DataFrame.
        """
        self.load_data()
        self.merge_datasets()
        self.save_processed_data()
        self.print_summary()
        return self.merged_df


if __name__ == "__main__":
    builder = CreditDatasetBuilder()
    df = builder.build()
