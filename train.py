import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import json

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# Fit the model
model = XGBRegressor(seed=42)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, verbose=True)

y_pred = model.predict(X_test)
R2_score = np.round(r2_score(y_test,y_pred)*100,2)

results = model.evals_result()

loss_df = pd.DataFrame({"Train Error":results['validation_0']['rmse'],
                        "Test Error":results['validation_1']['rmse']}).to_csv("Loss.csv", index=False)

with open("metrics.json", 'w') as f:
    json.dump({ "R2_Score": R2_score}, f)