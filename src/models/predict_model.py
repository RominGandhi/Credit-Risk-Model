import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


class ModelComparator:
    """Loads all trained models and compares their performance on the test set.

    Generates a side-by-side metrics table, a grouped bar chart comparing
    all metrics, and an overlay ROC curve plot. Selects the best model
    based on ROC AUC.
    """

    def __init__(
        self,
        data_dir="src/data/processed",
        model_dir="src/models/saved",
        plot_dir="reports/figures",
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.plot_dir = plot_dir

        self.models = {}
        self.X_test = None
        self.y_test = None
        self.results = {}

        os.makedirs(self.plot_dir, exist_ok=True)

    def load_test_data(self):
        """Load the held-out test set from CSV."""
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        target = "default"
        features = [c for c in test_df.columns if c != target]
        self.X_test = test_df[features].values
        self.y_test = test_df[target].values
        print(f"Test set loaded: {self.X_test.shape}")

    def load_models(self):
        """Load all three saved model files from disk."""
        model_files = {
            "Logistic Regression": "logistic_regression_best.joblib",
            "Random Forest": "random_forest_best.joblib",
            "XGBoost": "xgboost_model.joblib",
        }
        for name, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Loaded: {name}")
            else:
                print(f"Not found, skipping: {path}")

    def evaluate_all(self):
        """Run predictions for every loaded model and collect metrics."""
        print("\n--- Model Comparison ---")
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]

            self.results[name] = {
                "Accuracy": accuracy_score(self.y_test, y_pred),
                "Precision": precision_score(self.y_test, y_pred, zero_division=0),
                "Recall": recall_score(self.y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(self.y_test, y_pred, zero_division=0),
                "ROC AUC": roc_auc_score(self.y_test, y_proba),
            }

        df = pd.DataFrame(self.results).T
        print("\n" + df.to_string())
        return df

    def plot_metric_comparison(self):
        """Save a grouped bar chart comparing all metrics across models."""
        df = pd.DataFrame(self.results).T
        metrics = df.columns.tolist()
        models = df.index.tolist()
        x = np.arange(len(metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, model_name in enumerate(models):
            values = df.loc[model_name].values
            ax.bar(x + i * width, values, width, label=model_name)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        path = os.path.join(self.plot_dir, "model_comparison_bar.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def plot_roc_comparison(self):
        """Save an overlay ROC curve plot with all models on one axis."""
        colors = {
            "Logistic Regression": "darkorange",
            "Random Forest": "steelblue",
            "XGBoost": "green",
        }
        fig, ax = plt.subplots(figsize=(7, 6))

        for name, model in self.models.items():
            y_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            color = colors.get(name, "black")
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend(loc="lower right")

        path = os.path.join(self.plot_dir, "roc_comparison.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def select_best_model(self):
        """Identify and print the best model by ROC AUC."""
        best_name = max(self.results, key=lambda k: self.results[k]["ROC AUC"])
        best_auc = self.results[best_name]["ROC AUC"]
        best_acc = self.results[best_name]["Accuracy"]
        print(f"\nBest model: {best_name}")
        print(f"  ROC AUC:  {best_auc:.4f}")
        print(f"  Accuracy: {best_acc:.4f}")
        return best_name

    def run(self):
        """Execute the full comparison pipeline."""
        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        self.load_test_data()
        self.load_models()
        comparison_df = self.evaluate_all()

        self.plot_metric_comparison()
        self.plot_roc_comparison()
        best = self.select_best_model()

        print("=" * 60)
        return comparison_df, best


if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.run()
