from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42


def load_telco_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the IBM Telco customer churn dataset from OpenML."""
    dataset = fetch_openml(name="telco-customer-churn", version=1, as_frame=True, parser="auto")
    frame = dataset.frame.copy()
    frame["Churn"] = frame["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    features = frame.drop(columns=["Churn"])
    target = frame["Churn"]
    return features, target


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(exclude=["object", "string"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def build_models(x_train: pd.DataFrame) -> dict[str, ImbPipeline]:
    """Create RF and XGBoost classifiers using preprocess + SMOTE."""
    preprocessor = build_preprocessor(x_train)
    return {
        "Random Forest": ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "smote",
                    SMOTE(random_state=RANDOM_STATE),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "smote",
                    SMOTE(random_state=RANDOM_STATE),
                ),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        min_child_weight=2,
                        scale_pos_weight=1,
                        random_state=RANDOM_STATE,
                        eval_metric="auc",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def plot_roc_curve(models: dict[str, ImbPipeline], x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        probabilities = model.predict_proba(x_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        auc_score = roc_auc_score(y_test, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"{name} (AUC = {auc_score:.3f})")

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    axis.set_title("ROC Curve for Churn Prediction")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(models: dict[str, ImbPipeline], x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, len(models), figsize=(12, 4), constrained_layout=True)
    for axis, (name, model) in zip(np.atleast_1d(axes), models.items()):
        predictions = model.predict(x_test)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis, xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        axis.set_title(f"{name} Confusion Matrix")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
    figure.suptitle("Confusion Matrices")
    figure.savefig(output_dir / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_feature_importance(model: ImbPipeline, output_dir: Path) -> None:
    preprocessor = model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    top_features = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(10)
    fig, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_features, x="importance", y="feature", palette="viridis", ax=axis)
    axis.set_title("Top 10 Most Influential Features")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_models(models: dict[str, ImbPipeline], x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, probabilities)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "roc_auc": auc_score,
            }
        )
        print(f"\n{name}:")
        print(f"ROC-AUC = {auc_score:.4f}")
        print(classification_report(y_test, predictions, digits=4))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    plot_roc_curve(models, x_test, y_test, output_dir)
    plot_confusion_matrices(models, x_test, y_test, output_dir)
    plot_feature_importance(models["XGBoost"], output_dir)
    return metrics


def run_training(output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    features, target = load_telco_dataset()
    features.to_csv(output_dir / "telco_customer_churn.csv", index=False)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = build_models(x_train)
    metrics = evaluate_models(models, x_train, x_test, y_train, y_test, output_dir)
    print("\nModel summary:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved output files to: {output_dir.resolve()}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn prediction models using telco customer records.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated CSVs and plots (default: outputs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_training(arguments.output_dir)
