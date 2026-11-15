import joblib
import pandas as pd

artifact = joblib.load("RandomForest_model.pkl")

model = artifact["model"]
features = artifact["selected_features"]

X = pd.DataFrame([[0, 1, 0]], columns=features)
print(model.predict(X))
