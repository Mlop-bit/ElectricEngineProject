import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime


TRAIN_FILE = "train_cut_features.csv"
TEST_FILE = "test_cut_features.csv"

FEATURE_SETS = {
    "all_features": [
        "peak",
        "ptp",
        "rms",
        "dominant_frequency_hz",
        "wobble_frequency_hz",
        "wobble_strength",
    ],
    "basic_features": [
        "peak",
        "ptp",
        "rms",
    ],
    "frequency_features": [
        "dominant_frequency_hz",
        "wobble_frequency_hz",
        "wobble_strength",
    ],
}

all_columns = pd.read_csv(TRAIN_FILE, nrows=0).columns
mel_features = [c for c in all_columns if c.startswith("mel_")]
mfcc_features = [c for c in all_columns if c.startswith("mfcc_")]

FEATURE_SETS["mel_features"] = mel_features
FEATURE_SETS["mfcc_features"] = mfcc_features
FEATURE_SETS["six_plus_mfcc"] = FEATURE_SETS["all_features"] + mfcc_features
FEATURE_SETS["six_plus_mel_mfcc"] = (
    FEATURE_SETS["all_features"] + mel_features + mfcc_features
)

SELECTED_FEATURE_SET = sys.argv[1] if len(sys.argv) > 1 else "all_features"
FEATURES = FEATURE_SETS[SELECTED_FEATURE_SET]

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

test_data["environment"] = test_data["noise"]

X_train = train_data[FEATURES]
y_train = train_data["class"]
groups = train_data.groupby(["class", "file"]).ngroup()

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

X_test = test_data[FEATURES]
y_test = test_data["class"]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])

param_grid = {
    "knn__n_neighbors": list(range(1, 31)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train, groups=groups)

best_knn = grid_search.best_estimator_
predictions = best_knn.predict(X_test)

overall_accuracy = accuracy_score(y_test, predictions)

print("Selected feature set:", SELECTED_FEATURE_SET)
print("Features used:", FEATURES)
print()
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", round(grid_search.best_score_, 4))
print("Overall test accuracy:", round(overall_accuracy, 4))
print()

print("Classification report:")
print(classification_report(y_test, predictions))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print()

test_data = test_data.copy()
test_data["prediction"] = predictions

print("Accuracy by testing environment:")
for environment in sorted(test_data["environment"].unique()):
    subset = test_data[test_data["environment"] == environment]
    env_accuracy = accuracy_score(subset["class"], subset["prediction"])
    print(f"{environment}: {env_accuracy:.4f} ({len(subset)} samples)")

# Create a separate folder for this run.
run_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
results_dir = (
    Path(__file__).resolve().parent
    / "results"
    / f"knn_{SELECTED_FEATURE_SET}_{run_time}"
)
results_dir.mkdir(parents=True, exist_ok=False)

# Save the main settings and scores.
summary = [
    "Model: kNN",
    f"Feature set: {SELECTED_FEATURE_SET}",
    f"Features: {FEATURES}",
    f"Training file: {TRAIN_FILE}",
    f"Test file: {TEST_FILE}",
    f"Training rows: {len(train_data)}",
    f"Training recording groups: {groups.nunique()}",
    f"Test rows: {len(test_data)}",
    "Validation: 5-fold StratifiedGroupKFold",
    "Groups: class and file",
    "Shuffle: True; random_state: 42",
    f"Best parameters: {grid_search.best_params_}",
    f"Cross-validation accuracy: {grid_search.best_score_:.6f}",
    f"Overall test accuracy: {overall_accuracy:.6f}",
    "",
    "Classification report:",
    classification_report(y_test, predictions, zero_division=0),
]

(results_dir / "summary.txt").write_text(
    "\n".join(summary),
    encoding="utf-8",
)

# Save accuracy for each environment.
environment_results = []

for environment, subset in test_data.groupby("environment"):
    environment_results.append({
        "environment": environment,
        "samples": len(subset),
        "accuracy": accuracy_score(
            subset["class"],
            subset["prediction"],
        ),
    })

pd.DataFrame(environment_results).to_csv(
    results_dir / "environment_accuracy.csv",
    index=False,
)

# Save a labeled confusion matrix.
labels = best_knn.classes_

pd.DataFrame(
    confusion_matrix(y_test, predictions, labels=labels),
    index=labels,
    columns=labels,
).to_csv(
    results_dir / "confusion_matrix.csv",
    index_label="actual_class",
)

# Save each test prediction.
test_data[["class", "file", "noise", "prediction"]].to_csv(
    results_dir / "predictions.csv",
    index=False,
)

# Save all parameter-search results.
pd.DataFrame(grid_search.cv_results_).to_csv(
    results_dir / "cross_validation_results.csv",
    index=False,
)

print("\nResults saved to:", results_dir)

# Accuracy for each environment, split by actual class.
class_rows = []

for environment, env_subset in test_data.groupby("environment"):
    for actual_class in labels:
        class_subset = env_subset[env_subset["class"] == actual_class]

        if len(class_subset) == 0:
            continue

        correct = int((class_subset["prediction"] == actual_class).sum())

        row = {
            "environment": environment,
            "actual_class": actual_class,
            "samples": len(class_subset),
            "correct": correct,
            "accuracy": correct / len(class_subset),
        }

        for predicted_class in labels:
            row[f"predicted_{predicted_class}"] = int(
                (class_subset["prediction"] == predicted_class).sum()
            )

        class_rows.append(row)

environment_class_results = pd.DataFrame(class_rows)

environment_class_results.to_csv(
    results_dir / "environment_class_accuracy.csv",
    index=False,
)

print("\nAccuracy by environment and class:")
print(environment_class_results.to_string(index=False)) 