import sys
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, root_mean_squared_error
import joblib
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

TABLE_NAME = "requests"

context = {
    "df": None,
    "y": None,
    "target": None,
    "db_url": None,
    "re_run": False,
    "engine": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "raw_features": [],
    "selected_features": [],
    "transformers": {},
    "transform_log": [],
    "first_transform": True,
    "baseline_df": None,
    "model": None,
    "path": None,
    "metrics": {}
}

SCALERS = {
    "std": StandardScaler,
    "minmax": MinMaxScaler
}

ENCODERS = {
    "onehot": OneHotEncoder,
    "label": OrdinalEncoder
}

ALGORITHMS = {
    "RandomForest": RandomForestClassifier,
    "LinReg": LinearRegression,
    "LogReg": LogisticRegression
}

METRICS = {
    "acc": accuracy_score,
    "f1": f1_score,
    "pre": precision_score,
    "rec": recall_score,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error
}

def send_response(status, message, file_path="", eval_results=None, code=0):
    """Helper function, that produces JSON in the Rascal Response-ADT format"""
    if eval_results is None:
        eval_results = {}
    res = {
        "status": status,
        "message": message,
        "modelFilePath": file_path,
        "evalResults": eval_results,
        "code": code
    }
    print(json.dumps(res))
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            request = json.loads(line)
            cmd = request.get("cmd")
            
            if cmd == "LOAD":
                path = request["path"]
                context["target"] = request["target"]
                context["db_url"] = request["dbURL"] or None
                context["re_run"] = request["reRun"]

                if context["db_url"]:
                    try:
                        context["engine"] = create_engine(context["db_url"], pool_pre_ping=True)
                        with context["engine"].connect():
                            pass
                    except Exception as e:
                        send_response("ERROR", f"Database connection failed: {e}", code=4)
                        continue

                engine = context["engine"]

                if context["re_run"]:
                    if not inspect(engine).has_table(TABLE_NAME):
                        send_response("ERROR", f"re_run=true, but table '{TABLE_NAME}' does not exist.", code=5)
                        continue

                    full_df = pd.read_sql_table(TABLE_NAME, engine)
                    full_df.columns = [str(c) for c in full_df.columns]

                    if context["target"] not in full_df.columns:
                        send_response("ERROR", f"Specified target {context['target']} is not a column in table '{TABLE_NAME}'.",
                                      code=2)
                        continue

                    n_total = len(full_df)
                    full_df = full_df.dropna(subset=[context["target"]]).reset_index(drop=True)
                    t = full_df[context["target"]]
                    if pd.api.types.is_float_dtype(t) and (t % 1 == 0).all():
                        full_df[context["target"]] = t.astype("int64")
                    info = f"Loaded from database ({n_total} rows, {n_total - len(full_df)} without target dropped)."

                else:
                    if not os.path.exists(path):
                        send_response("ERROR", f"File not found: {path}", code=1)
                        continue

                    full_df = pd.read_csv(path)

                    if context["target"] not in full_df.columns:
                        send_response("ERROR", f"Specified target {context['target']} is not a column in loaded CSV.", code=2)
                        continue

                    if engine is not None:
                        full_df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
                        info = f"CSV loaded and written to table '{TABLE_NAME}'."
                    else:
                        info = "CSV loaded (no database configured)"

                context["y"] = full_df[context["target"]]
                context["df"] = full_df.drop(columns=[context["target"]])
                context["raw_features"] = list(context["df"].columns)
                
                send_response("SUCCESS", f"{info}. Form: {context['df'].shape}")
            
            elif cmd == "SPLIT":
                ratio = request["ratio"]
                random_state = int(request["randomState"])
                X = context["df"]
                y = context["y"]
                
                if X is None:
                    send_response("ERROR", "No data loaded. Split not possible.")
                    continue
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=random_state)
                context["X_train"] = X_train
                context["X_test"] = X_test
                context["y_train"] = y_train
                context["y_test"] = y_test
                
                send_response("SUCCESS", f"Data splitted (Train: {len(X_train)}, Test: {len(X_test)})")
            
            elif cmd == "SELECT":
                if context["re_run"]:
                    context["selected_features"] = list(context["raw_features"])
                    send_response("SUCCESS", f"re_run=true: SELECT skipped, using columns from database: {context['selected_features']}")
                    continue

                context["selected_features"] = request["features"]

                if context["X_train"] is not None:
                    missing = [f for f in context["selected_features"] if f not in context["X_train"].columns]
                    if missing:
                        send_response("ERROR", f"Columns not found in Dataframe: {missing}", code=3)
                        continue
                
                    context["X_train"] = context["X_train"][context["selected_features"]]
                    context["X_test"] = context["X_test"][context["selected_features"]]
                
                else:
                    if context["df"] is None:
                        send_response("ERROR", "No data loaded.")
                        continue

                    missing = [f for f in context["selected_features"] if f not in context["df"].columns]
                    if missing:
                        send_response("ERROR", f"Columns not found in Dataframe: {missing}", code=3)
                        continue
            
                    context["df"] = context["df"][context["selected_features"]]

                to_drop = []
                engine = context["engine"]
                if engine is not None:
                    db_cols = [c["name"] for c in inspect(engine).get_columns(TABLE_NAME)]
                    to_drop = [c for c in db_cols if c not in context["selected_features"] and c != context["target"]]
                    quote = engine.dialect.identifier_preparer.quote
                    try:
                        with engine.begin() as conn:
                            for col in to_drop:
                                conn.execute(text(f"ALTER TABLE {quote(TABLE_NAME)} DROP COLUMN {quote(col)}"))
                    except Exception as e:
                        send_response("ERROR", f"Dropping columns in database failed: {e}", code=6)
                        continue



                kept = context["selected_features"]
                msg = f"Features selected successfully. Kept columns: {kept}"
                if engine is not None:
                    msg += f" Dropped in database: {to_drop}"
                send_response("SUCCESS", msg)
            
            elif cmd == "TRANSFORM":
                action = request["action"]
                feature = request["feature"]
                method = request["method"]

                if context["first_transform"]:
                    source = context["X_train"] if context["X_train"] is not None else context["df"]
                    context["baseline_df"] = source.copy()
                    context["first_transform"] = False

                context["transform_log"] += [(action, feature, method)]
                
                working_on_split = context["X_train"] is not None
                
                if action == "scale":
                    scaler = SCALERS[method]()
                    if working_on_split:
                        if feature not in context["X_train"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                            continue
                        context["X_train"][[feature]] = scaler.fit_transform(context["X_train"][[feature]])
                        context["X_test"][[feature]] = scaler.transform(context["X_test"][[feature]])
                        context["transformers"][f"scaler_{feature}"] = scaler
                    else:
                        if feature not in context["df"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                            continue
                        context["df"][[feature]] = scaler.fit_transform(context["df"][[feature]])
                        context["transformers"][f"scaler_{feature}"] = scaler
                        
                elif action == "fillna":
                    imputer = SimpleImputer(strategy=method)
                    if working_on_split:
                        if feature not in context["X_train"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                            continue
                        context["X_train"][[feature]] = imputer.fit_transform(context["X_train"][[feature]])
                        context["X_test"][[feature]] = imputer.transform(context["X_test"][[feature]])
                        context["transformers"][f"imputer_{feature}"] = imputer
                    else:
                        if feature not in context["df"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                            continue
                        context["df"][[feature]] = imputer.fit_transform(context["df"][[feature]])
                        context["transformers"][f"imputer_{feature}"] = imputer
                        
                elif action == "encode":
                    if method == "onehot":
                        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        if working_on_split:
                            if feature not in context["X_train"].columns:
                                send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                                continue
                            train_encoded = ohe.fit_transform(context["X_train"][[feature]])
                            test_encoded = ohe.transform(context["X_test"][[feature]])                         
                            new_cols = ohe.get_feature_names_out([feature])                          
                            train_df_encoded = pd.DataFrame(train_encoded, columns=new_cols, index=context["X_train"].index)
                            test_df_encoded = pd.DataFrame(test_encoded, columns=new_cols, index=context["X_test"].index)                           
                            context["X_train"] = pd.concat([context["X_train"].drop(columns=[feature]), train_df_encoded], axis=1)
                            context["X_test"] = pd.concat([context["X_test"].drop(columns=[feature]), test_df_encoded], axis=1)
                            context["transformers"][f"encoder_{feature}"] = ohe
                        else:
                            if feature not in context["df"].columns:
                                send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                                continue
                            df_encoded = ohe.fit_transform(context["df"][[feature]])
                            new_cols = ohe.get_feature_names_out([feature])                        
                            df_encoded_pandas = pd.DataFrame(df_encoded, columns=new_cols, index=context["df"].index)                           
                            context["df"] = pd.concat([context["df"].drop(columns=[feature]), df_encoded_pandas], axis=1)                            
                            context["transformers"][f"encoder_{feature}"] = ohe        
                    else: #label
                        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        if working_on_split:
                            if feature not in context["X_train"].columns:
                                send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                                continue
                            context["X_train"][[feature]] = le.fit_transform(context["X_train"][[feature]])
                            context["X_test"][[feature]] = le.transform(context["X_test"][[feature]])
                            context["transformers"][f"encoder_{feature}"] = le
                        else:
                            if feature not in context["df"].columns:
                                send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                                continue
                            context["df"][[feature]] = le.fit_transform(context["df"][[feature]])
                            context["transformers"][f"encoder_{feature}"] = le 
                send_response("SUCCESS", f"Transformation {action}({method}) applied to {feature}.")

            elif cmd == "TRAIN":
                algo = request["algo"]
                hyperparams = request["hyperparameters"]
                model_dir = request["modelDir"]

                if context["first_transform"]:
                    source = context["X_train"] if context["X_train"] is not None else context["df"]
                    context["baseline_df"] = source.copy()
                
                if context["X_train"] is None:
                    context["X_train"] = context["df"]
                    context["y_train"] = context["y"]
                
                if algo not in ALGORITHMS:
                    send_response("ERROR", f"Algorithm {algo} not supported.")
                    continue

                typed_params = {}
                for k, v in hyperparams.items():
                    if v in ("true", "false"):
                        typed_params[k] = (v == "true")
                        continue
                    try:
                        typed_params[k] = int(v)
                    except ValueError:
                        try:
                            typed_params[k] = float(v)
                        except ValueError:
                            typed_params[k] = v

                try:
                    model = ALGORITHMS[algo](**typed_params)
                except TypeError as e:
                    send_response("ERROR", f"Hyperparameter included unexpected keyword: {str(e)}")
                    continue
                try:
                    model.fit(context["X_train"], context["y_train"])
                except ValueError as e:
                    send_response("ERROR",f"Model training failed: {str(e)}")
                    continue
                except TypeError as e:
                    send_response("ERROR",f"Hyperparameter has wrong type: {str(e)}")
                    continue
                context["model"] = model
                
                os.makedirs(model_dir, exist_ok=True)
                file_name = f"{algo}_model.pkl"
                model_file_path = Path(os.path.join(model_dir, file_name)).resolve()
                context["path"] = model_file_path
                deployment_artifact = {
                    "model": context["model"],
                    "transformers": context["transformers"],
                    "transform_log": context["transform_log"],
                    "selected_features": context["selected_features"] if context["selected_features"] else context["raw_features"],
                    "transformed_columns": list(context["X_train"].columns)
                }
                joblib.dump(deployment_artifact, model_file_path)
                
                send_response("SUCCESS", f"Model {algo} trained successfully.", str(model_file_path))
            
            elif cmd == "EVAL":
                metric = request["metric"]
                model = context["model"]

                eval_results = {}

                if context["X_test"] is not None:
                    y_pred = model.predict(context["X_test"])
                    result = METRICS[metric](context["y_test"], y_pred)
                    eval_results[metric] = result
                else:
                    y_pred = model.predict(context["X_train"])
                    result = METRICS[metric](context["y_train"], y_pred)
                    eval_results[metric] = result
                
                context["metrics"][metric] = result
                send_response("SUCCESS", f"Evaluated model with metric {metric}: {result}", eval_results=eval_results)

            elif cmd == "MONITOR":
                methods = request["methods"]
                features = request["features"]
                windows = request["windows"]
                frequencies = request["frequencies"]
                min_effects = request["minEffects"]
                thresholds = request["thresholds"]

                deployment_artifact = joblib.load(context["path"])
                deployment_artifact["target"] = context["target"]
                deployment_artifact["baseline_df"] = context["baseline_df"][features]
                deployment_artifact["methods"] = methods
                deployment_artifact["windows"] = windows
                deployment_artifact["check_every"] = frequencies
                deployment_artifact["min_effects"] = min_effects
                deployment_artifact["thresholds"] = thresholds
                deployment_artifact["db_url"] = context["db_url"]
                joblib.dump(deployment_artifact, context["path"])
                send_response("SUCCESS", f"Monitored features selected successfully: {features}")

            else:
                send_response("ERROR", f"Unknown command: {cmd}")
                
        except Exception as e:
            send_response("ERROR", f"Python exception: {str(e)}")

if __name__ == "__main__":
    main()