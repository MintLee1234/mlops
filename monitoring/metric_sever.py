from fastapi import FastAPI
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from starlette.responses import Response
import time
import pandas as pd
import numpy as np

app = FastAPI()

# --- Metrics cho model ---
PREDICTION_COUNT = Counter(
    "prediction_count", "Số lượng dự đoán model đã thực hiện"
)
PREDICTION_TIME = Histogram(
    "prediction_duration_seconds", "Thời gian xử lý mỗi lần dự đoán"
)

# --- Metrics cho dữ liệu ---
LABEL_DISTRIBUTION = Gauge(
    "label_distribution", "Tỉ lệ các nhãn trong dữ liệu", ["label"]
)
MISSING_COUNT = Gauge(
    "missing_value_count", "Số lượng giá trị thiếu trên mỗi cột", ["column"]
)

# --- Hàm mô phỏng kiểm tra dữ liệu ---
def monitor_data(X: pd.DataFrame, y: pd.Series):
    # Tính tỉ lệ nhãn
    total = len(y)
    if total > 0:
        for label in y.unique():
            count = (y == label).sum()
            LABEL_DISTRIBUTION.labels(label=str(label)).set(count / total)

    # Đếm missing value cho từng cột
    for col in X.columns:
        missing = X[col].isnull().sum()
        MISSING_COUNT.labels(column=col).set(missing)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/predict")
def predict():
    start = time.time()

    # --- Mô phỏng dữ liệu đầu vào ---
    data = {
        "feature1": [1.0, 2.5, np.nan, 4.1],
        "feature2": [0.5, 0.3, 0.2, 0.1],
    }
    X = pd.DataFrame(data)
    y = pd.Series([0, 1, 0, 1])  # mô phỏng nhãn

    # --- Gọi hàm theo dõi dữ liệu ---
    monitor_data(X, y)

    # --- Mô phỏng xử lý model ---
    time.sleep(0.5)
    PREDICTION_COUNT.inc()
    PREDICTION_TIME.observe(time.time() - start)

    return {"result": "OK"}
