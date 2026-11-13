import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import create_model

artifact = joblib.load("LogReg_model.pkl")
model = artifact["model"]
transformers = artifact["transformers"]
transform_log = artifact["transform_log"]
transformed_columns = artifact["transformed_columns"]
selected_features = artifact["selected_features"]

InputSchema = create_model("InputSchema", **{f: (float | str, ...) for f in selected_features})

app = FastAPI()


def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
   for action, feature, method in transform_log:
       key = f"{'scaler' if action=='scale' else 'imputer' if action=='fillna' else 'encoder'}_{feature}"
       tr = transformers[key]
       if action == "encode" and hasattr(tr, "get_feature_names_out"):
           encoded = tr.transform(df[[feature]])
           new_cols = tr.get_feature_names_out([feature])
           df = pd.concat([df.drop(columns=[feature]), pd.DataFrame(encoded, columns=new_cols, index=df.index)], axis=1)
       else:
           df[[feature]] = tr.transform(df[[feature]])
   return df.reindex(columns=transformed_columns, fill_value=0)


@app.post("/predict")
def predict(item: InputSchema):
   raw = item.model_dump()

   df = pd.DataFrame([raw])
   df = apply_transforms(df)
   pred = model.predict(df)[0]
   return {"prediction": pred.item() if hasattr(pred, "item") else pred}
    