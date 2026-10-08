"""Train and evaluate a small neural-network engine classifier."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "train_cut_features.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "engine_classifier.joblib"
FEATURE_COLUMNS = ["peak", "ptp", "rms", "dominant_frequency_hz"]
RANDOM_STATE = 42
CONTIGUOUS_GROUP_SIZE = 10


def groups_from_filenames(filenames: list[str]) -> np.ndarray:
    """Keep blocks of neighboring cuts together to reduce temporal leakage."""
    groups = []
    for filename in filenames:
        match = re.search(r"_(\d+)$", Path(str(filename)).stem)
        if match:
            cut_number = int(match.group(1))
            groups.append(f"cut-block-{cut_number // CONTIGUOUS_GROUP_SIZE}")
        else:
            groups.append(str(filename))
    return np.asarray(groups)


def train(csv_path: Path, model_path: Path, test_size: float) -> None:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

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
    groups = groups_from_filenames([row["file"] for row in rows])
    if not np.isfinite(x).all():
        raise ValueError("Feature CSV contains missing or non-finite feature values")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_indices, test_indices = next(splitter.split(x, y, groups))
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
    model.fit(x[train_indices], y[train_indices])
    predictions = model.predict(x[test_indices])
    labels = sorted(str(label) for label in np.unique(y))

    print(f"Rows: {len(rows)} ({len(train_indices)} train, {len(test_indices)} test)")
    print(f"Classes: {', '.join(labels)}")
    print(f"Test accuracy: {accuracy_score(y[test_indices], predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(
        y[test_indices], predictions, labels=labels, zero_division=0, digits=3
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"Labels: {labels}")
    print(confusion_matrix(y[test_indices], predictions, labels=labels))

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
                        help=f"feature CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"where to save the trained model (default: {DEFAULT_MODEL})")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="fraction of cut-number groups reserved for evaluation (default: 0.2)")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"feature CSV does not exist: {args.csv}; run scripts/extract_features.py first")
    train(args.csv, args.model, args.test_size)


if __name__ == "__main__":
    main()
