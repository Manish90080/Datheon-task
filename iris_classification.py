"""Benchmark Iris classification models and save reproducible visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def build_models(random_state: int) -> dict[str, Pipeline]:
    """Create scaled classification pipelines with reproducible estimators."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", DecisionTreeClassifier(random_state=random_state)),
            ]
        ),
    }


def save_pairplot(data: pd.DataFrame, output_dir: Path) -> None:
    plot = sns.pairplot(
        data,
        vars=[column for column in data.columns if column != "species"],
        hue="species",
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.75, "s": 28},
    )
    plot.figure.suptitle("Iris dataset — feature relationships", y=1.02)
    plot.savefig(output_dir / "iris_pairplot.png", dpi=160, bbox_inches="tight")
    plt.close(plot.figure)


def save_confusion_matrices(
    results: dict[str, dict[str, object]],
    class_names: list[str],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, len(results), figsize=(11, 4.5), constrained_layout=True)
    for axis, (name, result) in zip(np.atleast_1d(axes), results.items()):
        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
        )
        axis.set_title(name)
        axis.set_xlabel("Predicted species")
        axis.set_ylabel("True species")
        axis.tick_params(axis="x", rotation=25)
        axis.tick_params(axis="y", rotation=0)
    figure.suptitle("Test-set confusion matrices")
    figure.savefig(output_dir / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_decision_boundaries(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    class_names: list[str],
    random_state: int,
    output_dir: Path,
) -> None:
    """Plot separate two-feature models; benchmark scores use all four features."""
    feature_names = list(x_train.columns[:2])
    train_features = x_train[feature_names]
    test_features = x_test[feature_names]
    models = build_models(random_state)

    x_min = min(train_features.iloc[:, 0].min(), test_features.iloc[:, 0].min()) - 0.5
    x_max = max(train_features.iloc[:, 0].max(), test_features.iloc[:, 0].max()) + 0.5
    y_min = min(train_features.iloc[:, 1].min(), test_features.iloc[:, 1].min()) - 0.5
    y_max = max(train_features.iloc[:, 1].max(), test_features.iloc[:, 1].max()) + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 240),
        np.linspace(y_min, y_max, 240),
    )
    mesh = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=feature_names,
    )

    figure, axes = plt.subplots(1, len(models), figsize=(12, 5), constrained_layout=True)
    colors = ["#7abf8e", "#f0ca70", "#dc8b83"]
    for axis, (name, model) in zip(np.atleast_1d(axes), models.items()):
        model.fit(train_features, y_train)
        regions = model.predict(mesh).reshape(xx.shape)
        axis.contourf(
            xx,
            yy,
            regions,
            levels=np.arange(len(class_names) + 1) - 0.5,
            colors=colors,
            alpha=0.45,
        )
        for class_id, class_name in enumerate(class_names):
            train_mask = y_train == class_id
            test_mask = y_test == class_id
            axis.scatter(
                train_features.loc[train_mask].iloc[:, 0],
                train_features.loc[train_mask].iloc[:, 1],
                s=24,
                label=f"{class_name} (train)",
                edgecolors="white",
                linewidths=0.4,
            )
            axis.scatter(
                test_features.loc[test_mask].iloc[:, 0],
                test_features.loc[test_mask].iloc[:, 1],
                s=46,
                marker="x",
                color="#20251f",
                label=f"{class_name} (test)",
            )
        axis.set_title(f"{name} — 2-feature view")
        axis.set_xlabel(feature_names[0])
        axis.set_ylabel(feature_names[1])
        axis.legend(fontsize=7, loc="best", ncol=2)
    figure.suptitle("Illustrative decision boundaries (first two features)")
    figure.savefig(output_dir / "decision_boundaries.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_benchmark(output_dir: Path, test_size: float, random_state: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True)
    features = iris.data
    target = iris.target
    class_names = list(iris.target_names)

    dataset = features.copy()
    dataset["species"] = target.map(dict(enumerate(class_names)))
    dataset.to_csv(output_dir / "iris_dataset.csv", index=False)
    save_pairplot(dataset, output_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    results: dict[str, dict[str, object]] = {}
    metric_rows: list[dict[str, float | str]] = []
    for name, model in build_models(random_state).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = {
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=np.arange(len(class_names)),
            )
        }
        metric_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_weighted": precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "recall_weighted": recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "f1_weighted": f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
            }
        )
        report = classification_report(
            y_test,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
        print(f"\n{name} — classification report")
        print(report)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    save_confusion_matrices(results, class_names, output_dir)
    save_decision_boundaries(
        x_train,
        x_test,
        y_train,
        y_test,
        class_names,
        random_state,
        output_dir,
    )

    print(f"\nTrain/test rows: {len(x_train)}/{len(x_test)} ({1 - test_size:.0%}/{test_size:.0%})")
    print("\nWeighted test-set metrics")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved dataset, metrics, and visualizations to: {output_dir.resolve()}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Logistic Regression and Decision Tree on the Iris dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for the dataset, metrics, and plots (default: outputs).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset held out for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {RANDOM_STATE}).",
    )
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_benchmark(arguments.output_dir, arguments.test_size, arguments.random_state)
