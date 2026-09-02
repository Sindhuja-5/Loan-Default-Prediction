import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score

from preprocess import preprocess_data


df = pd.read_csv("data/test.csv")

y = df["loan_status"]

X = df.drop("loan_status", axis=1)

X = preprocess_data(X)

model = xgb.XGBClassifier()

model.load_model("model/final.model")

y_pred = model.predict_proba(X)[:, 1]

print(
    "ROC-AUC:",
    roc_auc_score(y, y_pred)
)