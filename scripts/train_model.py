"""Train and evaluate a small neural-network engine classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "train_cut_features.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "test_cut_features.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "engine_classifier.joblib"
FEATURE_COLUMNS = ["peak", "ptp", "rms", "dominant_frequency_hz"]
RANDOM_STATE = 42


def load_features(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate feature values and labels from a CSV file."""
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required = {"class", "file", *FEATURE_COLUMNS}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError(f"Feature CSV contains no rows: {csv_path}")

    x = np.asarray([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows])
    y = np.asarray([row["class"] for row in rows], dtype=str)
    if not np.isfinite(x).all():
        raise ValueError("Feature CSV contains missing or non-finite feature values")
    return x, y


def train(train_csv: Path, test_csv: Path, model_path: Path) -> None:
    x_train, y_train = load_features(train_csv)
    x_test, y_test = load_features(test_csv)
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

    print(f"Training rows from {train_csv.name}: {len(y_train)}")
    print(f"Test rows from {test_csv.name}: {len(y_test)}")
    print(f"Classes: {', '.join(labels)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
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
        "feature_columns": FEATURE_COLUMNS,
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
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"feature CSV does not exist: {args.csv}; run scripts/extract_features.py first")
    if not args.test_csv.is_file():
        parser.error(f"test feature CSV does not exist: {args.test_csv}; run scripts/extract_features.py first")
    train(args.csv, args.test_csv, args.model)


if __name__ == "__main__":
    main()
