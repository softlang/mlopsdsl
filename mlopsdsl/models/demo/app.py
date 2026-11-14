import os
import time
import logging
import threading
from collections import deque

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from pydantic import create_model
from pandas.api import types as pdt
from scipy.stats import ks_2samp, chisquare
from sqlalchemy import create_engine, MetaData, Table, insert
from sqlalchemy.engine import make_url

logger = logging.getLogger("uvicorn.error")

artifact = joblib.load("LogReg_model.pkl")
model = artifact["model"]
transformers = artifact["transformers"]
transform_log = artifact["transform_log"]
transformed_columns = artifact["transformed_columns"]
selected_features = artifact["selected_features"]
target = artifact["target"]
baseline_df = artifact["baseline_df"]
methods = artifact["methods"]
windows = artifact["windows"]
thresholds = artifact["thresholds"]
check_every = artifact["check_every"]
min_effects = artifact["min_effects"]

TABLE_NAME = "requests"
LATENCY_LIMIT_MS = 500  # None = no latency check

CHI_ALPHA = 0.5

url = make_url(artifact["db_url"])
override = os.environ.get("DB_HOST_OVERRIDE")
if override and url.host in ("localhost", "127.0.0.1"):
    url = url.set(host=override)

engine = create_engine(url, pool_pre_ping=True)
requests_table = Table(TABLE_NAME, MetaData(), autoload_with=engine)


def store_request(payload: dict, pred) -> None:
    try:
        row = {f: payload[f] for f in selected_features}
        row[target] = pred.item() if hasattr(pred, "item") else pred
        with engine.begin() as conn:
            conn.execute(insert(requests_table), row)
    except Exception:
        logger.exception("Saving request input in database failed")


def python_type_for(series: pd.Series):
    dtype = series.dtype
    if pdt.is_bool_dtype(dtype):
        return bool
    if pdt.is_integer_dtype(dtype):
        return int
    if pdt.is_numeric_dtype(dtype):
        return float
    return str


def type_for_feature(feature: str):
    if feature in baseline_df.columns:
        return python_type_for(baseline_df[feature])
    return float | str


InputSchema = create_model(
    "InputSchema",
    **{f: (type_for_feature(f), ...) for f in selected_features},
)

drift_config = {}
for feature, window, method, threshold, every, min_effect in zip(
    baseline_df.columns, windows, methods, thresholds, check_every, min_effects
):
    clean = baseline_df[feature].dropna()
    cfg = {
        "window": window,
        "method": method,
        "threshold": threshold,
        "check_every": max(int(every), 1),
        "min_effect": float(min_effect),
    }
    if method == "KS":
        cfg["baseline"] = clean.to_numpy(dtype=float)
    else:
        cfg["base_counts"] = clean.value_counts()
    drift_config[feature] = cfg

drift_windows = {f: deque(maxlen=cfg["window"]) for f, cfg in drift_config.items()}
since_check = {f: 0 for f in drift_config}
state_lock = threading.Lock()  # protects drift_windows and since_check

app = FastAPI()


def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    for action, feature, method in transform_log:
        key = f"{'scaler' if action == 'scale' else 'imputer' if action == 'fillna' else 'encoder'}_{feature}"
        tr = transformers[key]
        if action == "encode" and hasattr(tr, "get_feature_names_out"):
            encoded = tr.transform(df[[feature]])
            new_cols = tr.get_feature_names_out([feature])
            df = pd.concat(
                [df.drop(columns=[feature]), pd.DataFrame(encoded, columns=new_cols, index=df.index)],
                axis=1,
            )
        else:
            df[[feature]] = tr.transform(df[[feature]])
    return df.reindex(columns=transformed_columns, fill_value=0)


def ks_test(feature: str, window_values: list):
    cur = pd.Series(window_values, dtype=float).dropna().to_numpy()
    base = drift_config[feature]["baseline"]
    if len(cur) == 0 or len(base) == 0:
        return None
    res = ks_2samp(base, cur)
    stat, p = float(res.statistic), float(res.pvalue)
    return stat, p, stat  # for KS, D itself is the effect size


def chi2_test(feature: str, window_values: list):
    base_counts = drift_config[feature]["base_counts"]
    cur = pd.Series(window_values).dropna()
    n = len(cur)
    if n == 0 or base_counts.empty:
        return None

    cats = base_counts.index.union(pd.Index(cur.unique()))
    observed = cur.value_counts().reindex(cats, fill_value=0)

    smoothed = base_counts.reindex(cats, fill_value=0) + CHI_ALPHA
    expected = smoothed / smoothed.sum() * n

    res = chisquare(f_obs=observed.to_numpy(), f_exp=expected.to_numpy())
    stat, p = float(res.statistic), float(res.pvalue)
    effect = float(np.sqrt(stat / n))  # Cohen's w
    return stat, p, effect


def compute_drift(method: str, feature: str, window_values: list):
    return ks_test(feature, window_values) if method == "KS" else chi2_test(feature, window_values)


@app.post("/predict")
def predict(item: InputSchema, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    raw = item.model_dump()
    warnings = []

    to_check = {}
    with state_lock:
        for feature, cfg in drift_config.items():
            win = drift_windows[feature]
            win.append(raw[feature])
            since_check[feature] += 1
            if len(win) == cfg["window"] and since_check[feature] >= cfg["check_every"]:
                since_check[feature] = 0
                to_check[feature] = list(win)

    for feature, values in to_check.items():
        cfg = drift_config[feature]
        method, alpha = cfg["method"], cfg["threshold"]
        try:
            result = compute_drift(method, feature, values)
        except Exception:
            logger.exception("Drift test failed for feature %s", feature)
            continue
        if result is None:
            continue
        stat, p_value, effect = result
        if p_value <= alpha and effect >= cfg["min_effect"]:
            warnings.append(
                f"Drift detected on {feature} ({method}): "
                f"p-value={p_value:.4g} < alpha={alpha}, statistic={stat:.4f}, effect={effect:.4f}"
            )

    df = pd.DataFrame([raw])
    df = apply_transforms(df)
    pred = model.predict(df)[0]

    elapsed_ms = (time.perf_counter() - start) * 1000
    if LATENCY_LIMIT_MS is not None and elapsed_ms > LATENCY_LIMIT_MS:
        warnings.append(f"Latency {elapsed_ms:.1f}ms exceeded {LATENCY_LIMIT_MS}ms")

    background_tasks.add_task(store_request, raw, pred)

    result = {"prediction": pred.item() if hasattr(pred, "item") else pred}
    if warnings:
        result["warnings"] = warnings
    return result
    