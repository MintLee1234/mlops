# serve_model.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import uvicorn
import pandas as pd

with open('/home/minhle/mlops/last_best_run_id.txt') as f:
    best_run_id = f.readlines()[-1].strip().split(' - ')[-1]

model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

app = FastAPI()


class Input(BaseModel):
    data: list  # list of dicts


@app.post("/predict")
def predict(input: Input):
    df = pd.DataFrame(input.data)
    pred = model.predict(df)
    return {"predictions": pred.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)