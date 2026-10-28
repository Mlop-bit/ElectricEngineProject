"""Train and evaluate a small neural-network engine classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "train_cut_features.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "test_cut_features.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "engine_classifier.joblib"
METADATA_COLUMNS = {"class", "file", "noise"}
RANDOM_STATE = 42


def load_features(
    csv_path: Path,
    feature_columns: list[str] | None = None,
    noise_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and validate feature values and labels from a CSV file."""
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if feature_columns is None:
        feature_columns = [
            name for name in fieldnames if name not in METADATA_COLUMNS
        ]
    if not feature_columns:
        raise ValueError(f"Feature CSV contains no feature columns: {csv_path}")
    required = {"class", "file", *feature_columns}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {', '.join(sorted(missing))}")
    if noise_type is not None:
        if "noise" not in fieldnames:
            raise ValueError(f"Feature CSV has no 'noise' column: {csv_path}")
        rows = [row for row in rows if row["noise"] == noise_type]
        if not rows:
            raise ValueError(f"No rows with noise type '{noise_type}' in {csv_path}")
    if not rows:
        raise ValueError(f"Feature CSV contains no rows: {csv_path}")

    x = np.asarray([[float(row[column]) for column in feature_columns] for row in rows])
    y = np.asarray([row["class"] for row in rows], dtype=str)
    if not np.isfinite(x).all():
        raise ValueError("Feature CSV contains missing or non-finite feature values")
    return x, y, feature_columns


def train(
    train_csv: Path, test_csv: Path, model_path: Path,
    test_noise: str | None = None,
) -> None:
    x_train, y_train, feature_columns = load_features(train_csv)
    x_test, y_test, _ = load_features(test_csv, feature_columns, test_noise)
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="lbfgs",
            alpha=0.01,
            max_iter=1500,
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(str(label) for label in np.union1d(y_train, y_test))
    importance = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="accuracy",
        n_repeats=10,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows from {train_csv.name}: {len(y_train)}")
    print(f"Test rows from {test_csv.name}: {len(y_test)}")
    if test_noise:
        print(f"Test noise type: {test_noise}")
    print(f"Classes: {', '.join(labels)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\n=== FEATURE IMPORTANCE (permutation on test set) ===", flush=True)
    print(f"Using {len(feature_columns)} features.")
    print("Each # represents the relative accuracy drop when that feature is shuffled.")
    max_importance = max(abs(value) for value in importance.importances_mean)
    for index in np.argsort(importance.importances_mean)[::-1]:
        value = importance.importances_mean[index]
        bar_length = round(abs(value) / max_importance * 30) if max_importance else 0
        bar = ("#" * bar_length) if value >= 0 else ("<" * bar_length)
        if not bar:
            bar = "(no measurable drop)"
        print(
            f"{feature_columns[index]:<26} {value:+.4f}  "
            f"{bar}  (+/- {importance.importances_std[index]:.4f})"
        )
    print("===================================================\n")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=labels, zero_division=0, digits=3
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"Labels: {labels}")
    print(confusion_matrix(y_test, predictions, labels=labels))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": feature_columns,
        "classes": labels,
    }, model_path)
    print(f"Model saved to: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"training feature CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV,
                        help=f"test feature CSV (default: {DEFAULT_TEST_CSV})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"where to save the trained model (default: {DEFAULT_MODEL})")
    parser.add_argument("--test-noise", type=str,
                        help="evaluate only test rows with this noise type (from the CSV noise column)")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"feature CSV does not exist: {args.csv}; run scripts/extract_features.py first")
    if not args.test_csv.is_file():
        parser.error(f"test feature CSV does not exist: {args.test_csv}; run scripts/extract_features.py first")
    train(args.csv, args.test_csv, args.model, args.test_noise)


if __name__ == "__main__":
    main()
