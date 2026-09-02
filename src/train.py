import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from preprocess import preprocess_data


df = pd.read_csv("data/train.csv")

y = df["loan_status"]

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(6, 4))
y.value_counts().sort_index().plot(kind="bar")

plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Loans")
plt.tight_layout()

plt.savefig("results/class_distribution.png")
plt.show()

X = df.drop("loan_status", axis=1)

X = preprocess_data(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))

importance.sort_values().plot(
    kind="barh"
)

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()

plt.savefig("results/feature_importance.png")
plt.show()


# ============================================================
# EVALUATION
# ============================================================

y_pred_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)

print(
    "ROC-AUC:", roc_auc
)

model.save_model("model/final.model")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig("results/roc_curve.png")
plt.show()