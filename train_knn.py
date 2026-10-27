import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

SELECTED_FEATURE_SET = "all_features"
FEATURES = FEATURE_SETS[SELECTED_FEATURE_SET]

def extract_environment(filename):
    name = filename.replace(".wav", "").lower()

    if name.startswith("whitenoise"):
        return "whitenoise"
    elif name.startswith("atmo_"):
        parts = name.split("_")
        if len(parts) >= 2:
            return f"atmo_{parts[1]}"
        return "atmo"
    elif name.startswith("talking"):
        return "talking"
    elif name.startswith("stresstest"):
        return "stresstest"
    elif name.startswith("pure"):
        return "pure"
    else:
        return "unknown"

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

test_data["environment"] = test_data["file"].apply(extract_environment)

X_train = train_data[FEATURES]
y_train = train_data["class"]

X_test = test_data[FEATURES]
y_test = test_data["class"]

param_grid = {
    "n_neighbors": list(range(1, 31)),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

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