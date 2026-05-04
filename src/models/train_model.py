import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


class RandomForestModel:
    """Random Forest classifier for credit default prediction.

    Uses SMOTE oversampling with class_weight='balanced' to handle
    class imbalance. Hyperparameters are tuned via RandomizedSearchCV.
    """

    def __init__(
        self,
        data_dir="src/data/processed",
        model_dir="src/models/saved",
        plot_dir="reports/figures",
        random_state=42,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.plot_dir = plot_dir
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        self.best_model = None
        self.search_results = None

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def load_data(self):
        """Load preprocessed train and test sets from CSV files."""
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))

        target = "default"
        self.feature_names = [c for c in train_df.columns if c != target]

        self.X_train = train_df[self.feature_names].values
        self.y_train = train_df[target].values
        self.X_test = test_df[self.feature_names].values
        self.y_test = test_df[target].values

        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        print(f"Train default rate: {self.y_train.mean():.4f}")
        print(f"Test  default rate: {self.y_test.mean():.4f}")

    def train(self, n_iter=15, cv=5):
        """Run RandomizedSearchCV with SMOTE oversampling in a pipeline."""
        print("\n--- Hyperparameter Tuning (RandomizedSearchCV) ---")

        param_dist = {
            "rf__n_estimators": [100, 200, 300, 500],
            "rf__max_depth": [10, 20, 30, None],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__max_features": ["sqrt", "log2"],
        }

        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=self.random_state)),
            ("rf", RandomForestClassifier(
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring="f1",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(self.X_train, self.y_train)

        self.best_model = search.best_estimator_
        self.search_results = search

        print(f"\nBest params: {search.best_params_}")
        print(f"Best CV F1:  {search.best_score_:.4f}")

    def cross_validate(self, cv=5):
        """Run 5-fold cross-validation and report multiple metrics."""
        print("\n--- 5-Fold Cross-Validation ---")
        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        for metric in scoring_metrics:
            scores = cross_val_score(
                self.best_model, self.X_train, self.y_train,
                cv=cv, scoring=metric, n_jobs=-1,
            )
            print(f"  {metric:>10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    def evaluate(self):
        """Evaluate the best model on the held-out test set."""
        print("\n--- Test Set Evaluation ---")
        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_proba)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")

        print(f"\n{classification_report(self.y_test, y_pred, target_names=['No Default', 'Default'])}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    def plot_confusion_matrix(self):
        """Generate and save the confusion matrix heatmap."""
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Default", "Default"],
                    yticklabels=["No Default", "Default"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Random Forest: Confusion Matrix")

        path = os.path.join(self.plot_dir, "rf_confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def plot_roc_curve(self):
        """Generate and save the ROC curve plot."""
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = roc_auc_score(self.y_test, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Random Forest: ROC Curve")
        ax.legend(loc="lower right")

        path = os.path.join(self.plot_dir, "rf_roc_curve.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def plot_feature_importance(self, top_n=20):
        """Generate and save the top-N feature importance bar chart."""
        rf = self.best_model.named_steps["rf"]
        importances = rf.feature_importances_

        feat_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        top = feat_imp.head(top_n)

        fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Random Forest: Top {top_n} Feature Importances")

        path = os.path.join(self.plot_dir, "rf_feature_importance.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

        print(f"\nTop {top_n} features:")
        print(top.to_string(index=False))

    def save_model(self):
        """Persist the best model to disk with joblib."""
        path = os.path.join(self.model_dir, "random_forest_best.joblib")
        joblib.dump(self.best_model, path)
        print(f"\nModel saved to {path}")

    def run(self):
        """Execute the full Random Forest training pipeline."""
        print("=" * 60)
        print("RANDOM FOREST TRAINING PIPELINE")
        print("=" * 60)

        self.load_data()
        self.train()
        self.cross_validate()
        metrics = self.evaluate()

        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_feature_importance()
        self.save_model()

        print("\n" + "=" * 60)
        if metrics["accuracy"] >= 0.80:
            print(f"TARGET MET! Accuracy: {metrics['accuracy']:.4f}")
        else:
            print(f"TARGET NOT MET! Accuracy: {metrics['accuracy']:.4f} (need 0.80+)")
        print("=" * 60)

        return metrics


class LogisticRegressionModel:
    """Logistic Regression model for credit default prediction.

    Uses SMOTE oversampling and searches over L1/L2 penalties,
    regularization strength (C), solvers, and class weighting
    via GridSearchCV optimized on accuracy.
    """

    def __init__(
        self,
        data_dir="src/data/processed",
        model_dir="src/models/saved",
        plot_dir="reports/figures",
        random_state=42,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.plot_dir = plot_dir
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.best_model = None
        self.search_results = None

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def load_data(self):
        """Load preprocessed train and test sets from CSV files."""
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))

        target = "default"
        self.feature_names = [c for c in train_df.columns if c != target]

        self.X_train = train_df[self.feature_names].values
        self.y_train = train_df[target].values
        self.X_test = test_df[self.feature_names].values
        self.y_test = test_df[target].values

        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        print(f"Train default rate: {self.y_train.mean():.4f}")
        print(f"Test  default rate: {self.y_test.mean():.4f}")

    def train(self, cv=5):
        """Run GridSearchCV with SMOTE oversampling in a pipeline."""
        print("\n--- Hyperparameter Tuning (GridSearchCV) ---")

        param_grid = {
            "lr__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "lr__penalty": ["l1", "l2"],
            "lr__solver": ["liblinear", "saga"],
            "lr__class_weight": ["balanced", None],
        }

        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=self.random_state)),
            ("lr", LogisticRegression(
                max_iter=5000,
                random_state=self.random_state,
            )),
        ])

        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        search.fit(self.X_train, self.y_train)

        self.best_model = search.best_estimator_
        self.search_results = search

        print(f"\nBest params: {search.best_params_}")
        print(f"Best CV Accuracy: {search.best_score_:.4f}")

    def cross_validate(self, cv=5):
        """Run 5-fold cross-validation and report multiple metrics."""
        print("\n--- 5-Fold Cross-Validation ---")
        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        for metric in scoring_metrics:
            scores = cross_val_score(
                self.best_model, self.X_train, self.y_train,
                cv=cv, scoring=metric, n_jobs=-1,
            )
            print(f"  {metric:>10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    def evaluate(self):
        """Evaluate the best model on the held-out test set."""
        print("\n--- Test Set Evaluation ---")
        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_proba)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")

        print(f"\n{classification_report(self.y_test, y_pred, target_names=['No Default', 'Default'])}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    def plot_confusion_matrix(self):
        """Generate and save the confusion matrix heatmap."""
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                    xticklabels=["No Default", "Default"],
                    yticklabels=["No Default", "Default"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Logistic Regression: Confusion Matrix")

        path = os.path.join(self.plot_dir, "lr_confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def plot_roc_curve(self):
        """Generate and save the ROC curve plot."""
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = roc_auc_score(self.y_test, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Logistic Regression: ROC Curve")
        ax.legend(loc="lower right")

        path = os.path.join(self.plot_dir, "lr_roc_curve.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def save_model(self):
        """Persist the best model to disk with joblib."""
        path = os.path.join(self.model_dir, "logistic_regression_best.joblib")
        joblib.dump(self.best_model, path)
        print(f"\nModel saved to {path}")

    def run(self):
        """Execute the full Logistic Regression training pipeline."""
        print("=" * 60)
        print("LOGISTIC REGRESSION TRAINING PIPELINE")
        print("=" * 60)

        self.load_data()
        self.train()
        self.cross_validate()
        metrics = self.evaluate()

        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.save_model()

        print("\n" + "=" * 60)
        if metrics["accuracy"] >= 0.80:
            print(f"TARGET MET! Accuracy: {metrics['accuracy']:.4f}")
        else:
            print(f"TARGET NOT MET! Accuracy: {metrics['accuracy']:.4f} (need 0.80+)")
        print("=" * 60)

        return metrics


class XGBoostModel:
    """XGBoost classifier for credit default prediction.

    Uses scale_pos_weight to handle class imbalance and tunes
    hyperparameters via RandomizedSearchCV.
    """

    def __init__(
        self,
        data_dir="src/data/processed",
        model_dir="src/models/saved",
        plot_dir="reports/figures",
        random_state=42,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.plot_dir = plot_dir
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.best_model = None

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def load_data(self):
        """Load preprocessed train and test sets from CSV files."""
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))

        target = "default"
        self.feature_names = [c for c in train_df.columns if c != target]

        self.X_train = train_df[self.feature_names].values
        self.y_train = train_df[target].values
        self.X_test = test_df[self.feature_names].values
        self.y_test = test_df[target].values

        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        print(f"Train default rate: {self.y_train.mean():.4f}")
        print(f"Test  default rate: {self.y_test.mean():.4f}")

    def train(self, n_iter=20, cv=5):
        """Run RandomizedSearchCV with scale_pos_weight for class imbalance."""
        print("\n--- Hyperparameter Tuning (RandomizedSearchCV) ---")

        neg = len(self.y_train[self.y_train == 0])
        pos = len(self.y_train[self.y_train == 1])
        scale = neg / pos

        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        model = XGBClassifier(
            scale_pos_weight=scale,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring="f1",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(self.X_train, self.y_train)

        self.best_model = search.best_estimator_

        print(f"\nBest params: {search.best_params_}")
        print(f"Best CV F1: {search.best_score_:.4f}")

    def cross_validate(self, cv=5):
        """Run 5-fold cross-validation and report multiple metrics."""
        print("\n--- 5-Fold Cross-Validation ---")
        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        for metric in scoring_metrics:
            scores = cross_val_score(
                self.best_model, self.X_train, self.y_train,
                cv=cv, scoring=metric, n_jobs=-1,
            )
            print(f"  {metric:>10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    def evaluate(self):
        """Evaluate the best model on the held-out test set."""
        print("\n--- Test Set Evaluation ---")
        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_proba)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")

        print(f"\n{classification_report(self.y_test, y_pred, target_names=['No Default', 'Default'])}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    def plot_confusion_matrix(self):
        """Generate and save the confusion matrix heatmap."""
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["No Default", "Default"],
                    yticklabels=["No Default", "Default"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("XGBoost: Confusion Matrix")

        path = os.path.join(self.plot_dir, "xgb_confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def plot_roc_curve(self):
        """Generate and save the ROC curve plot."""
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = roc_auc_score(self.y_test, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="green", lw=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("XGBoost: ROC Curve")
        ax.legend(loc="lower right")

        path = os.path.join(self.plot_dir, "xgb_roc_curve.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def plot_feature_importance(self, top_n=20):
        """Generate and save the top-N feature importance bar chart."""
        importances = self.best_model.feature_importances_

        feat_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        top = feat_imp.head(top_n)

        fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="green")
        ax.set_xlabel("Importance")
        ax.set_title(f"XGBoost: Top {top_n} Feature Importances")

        path = os.path.join(self.plot_dir, "xgb_feature_importance.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def save_model(self):
        """Persist the best model to disk with joblib."""
        path = os.path.join(self.model_dir, "xgboost_model.joblib")
        joblib.dump(self.best_model, path)
        print(f"\nModel saved to {path}")

    def run(self):
        """Execute the full XGBoost training pipeline."""
        print("=" * 60)
        print("XGBOOST TRAINING PIPELINE")
        print("=" * 60)

        self.load_data()
        self.train()
        self.cross_validate()
        metrics = self.evaluate()

        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_feature_importance()
        self.save_model()

        print("\n" + "=" * 60)
        if metrics["accuracy"] >= 0.80:
            print(f"TARGET MET! Accuracy: {metrics['accuracy']:.4f}")
        else:
            print(f"TARGET NOT MET! Accuracy: {metrics['accuracy']:.4f} (need 0.80+)")
        print("=" * 60)

        return metrics


if __name__ == "__main__":
    print("\n>>> RANDOM FOREST <<<\n")
    rf = RandomForestModel()
    rf_metrics = rf.run()

    print("\n>>> LOGISTIC REGRESSION <<<\n")
    lr = LogisticRegressionModel()
    lr_metrics = lr.run()

    print("\n>>> XGBOOST <<<\n")
    xgb = XGBoostModel()
    xgb_metrics = xgb.run()
