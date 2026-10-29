"""
Basil's Model Training Script
Support Vector Machine (SVM) & Decision Trees
For Electric Engine Condition Classification
"""
# her all the necessary imports
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
except ImportError as exc:
    plt = None
    ConfusionMatrixDisplay = None
    print(f"\n Visualization disabled: {exc}")

# ============================================================
# 1. DATA INGESTION
# ============================================================
print("=" * 60)
print(" STARTING BASIL'S MODEL PIPELINE")
print("=" * 60)

# Locate the CSV file in the project root.
project_dir = Path(__file__).resolve().parent
csv_path = project_dir / "train_cut_features.csv"

print(f"\n Loading data from: {csv_path}")
df = pd.read_csv(csv_path)
print(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n Class distribution:")
print(df['class'].value_counts())

# ============================================================
# 2. DATA PREPARATION
# ============================================================
# Features (X) and Target (y)
# Dropping 'file' and 'noise' columns as they are not predictive features
X = df.drop(columns=['class', 'file', 'noise'])
y = df['class']

print(f"\n Features: {list(X.columns)}")
print(f" Target classes: {list(y.unique())}")

# ============================================================
# 3. DATA SEGREGATION (Train/Test Split)
# ============================================================
# 80% training, 20% testing
# stratify=y ensures equal class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n  Data split:")
print(f"   Training data: {X_train.shape[0]} rows")
print(f"   Test data:     {X_test.shape[0]} rows")

# ============================================================
# 4. FEATURE SCALING (Crucial for SVM!)
# ============================================================
# Standardizing data (mean=0, std=1)
# Required for distance-based algorithms like SVM and kNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn from training data only
X_test_scaled = scaler.transform(X_test)        # Apply same transformation to test data

print(f"\n Data scaled with StandardScaler")
print(f"   Feature means after scaling: {X_train_scaled.mean(axis=0).round(2)}")

# ============================================================
# 5. MODEL TRAINING
# ============================================================
print("\n" + "=" * 60)
print(" TRAINING MODELS")
print("=" * 60)

# Model 1: Support Vector Machine (SVM)
# RBF kernel handles non-linear boundaries well
print("\n Training Support Vector Machine (SVM)...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    random_state=42,
    probability=True
)
svm_model.fit(X_train_scaled, y_train)
print("    SVM model trained!")

# Model 2: Decision Tree
# max_depth limits the tree to prevent overfitting
print("\n Training Decision Tree...")
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
dt_model.fit(X_train_scaled, y_train)
print("    Decision Tree model trained!")

# ============================================================
# 6. MODEL EVALUATION
# ============================================================
def evaluate_model(model, model_name, X_test, y_test):
    """Evaluates a model and returns predictions"""
    print(f"\n{'=' * 60}")
    print(f" EVALUATION: {model_name}")
    print(f"{'=' * 60}")

    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    return y_pred, cm, accuracy

# Evaluate both models
svm_pred, svm_cm, svm_acc = evaluate_model(
    svm_model, "Support Vector Machine (SVM)", X_test_scaled, y_test
)

dt_pred, dt_cm, dt_acc = evaluate_model(
    dt_model, "Decision Tree", X_test_scaled, y_test
)

# ============================================================
# 7. VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print(" VISUALIZING RESULTS")
print("=" * 60)

if plt is not None and ConfusionMatrixDisplay is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SVM Confusion Matrix
    disp_svm = ConfusionMatrixDisplay(
        confusion_matrix=svm_cm,
        display_labels=svm_model.classes_
    )
    disp_svm.plot(ax=axes[0], cmap='Blues')
    axes[0].set_title(f'SVM Confusion Matrix\nAccuracy: {svm_acc*100:.2f}%',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Class')
    axes[0].set_ylabel('Actual Class')

    # Decision Tree Confusion Matrix
    disp_dt = ConfusionMatrixDisplay(
        confusion_matrix=dt_cm,
        display_labels=dt_model.classes_
    )
    disp_dt.plot(ax=axes[1], cmap='Greens')
    axes[1].set_title(f'Decision Tree Confusion Matrix\nAccuracy: {dt_acc*100:.2f}%',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('Actual Class')

    plt.tight_layout()
    plt.savefig(project_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n️  Confusion Matrix saved as: model_comparison.png")
    plt.show()
else:
    print("\n ℹ  Skipping plots because Matplotlib is blocked by Windows policy.")

# Decision-tree feature importance
feature_importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\n Decision Tree feature importance:")
for feature_name, importance in feature_importance.items():
    print(f"   {feature_name:<25} {importance:.4f}")

if plt is not None:
    importance_plot, importance_axis = plt.subplots(figsize=(8, 5))
    feature_importance.sort_values().plot.barh(
        ax=importance_axis,
        color="orange",
    )
    importance_axis.set_xlabel("Importance score")
    importance_axis.set_title("Feature Importance (Decision Tree)")
    importance_plot.tight_layout()
    importance_plot.savefig(
        project_dir / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# ============================================================
# 8. COMPARISON
# ============================================================
print("\n" + "=" * 60)
print(" MODEL COMPARISON")
print("=" * 60)
print(f"\n{'Model':<25} {'Accuracy':<10}")
print(f"{'-' * 35}")
print(f"{'Support Vector Machine':<25} {svm_acc*100:.2f}%")
print(f"{'Decision Tree':<25} {dt_acc*100:.2f}%")

best_model = "SVM" if svm_acc > dt_acc else "Decision Tree"
print(f"\n✨ Best model: {best_model}")

# ============================================================
# 9. SAVE MODELS
# ============================================================
print("\n" + "=" * 60)
print(" SAVING MODELS")
print("=" * 60)

models_dir = project_dir / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(svm_model, models_dir / "svm_model.pkl")
joblib.dump(dt_model, models_dir / "decision_tree_model.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")

print(f" Models saved in: {models_dir}")
print("   - svm_model.pkl")
print("   - decision_tree_model.pkl")
print("   - scaler.pkl")

print("\n" + "=" * 60)
print(" DONE! All models trained and evaluated.")
print("=" * 60)