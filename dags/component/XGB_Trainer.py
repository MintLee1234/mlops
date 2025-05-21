import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
import optuna
import mlflow
from datetime import datetime
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/minhle/mlops/mlops-postgresql-6a3c27b9fd84.json"


class XGB_Trainer:
    def __init__(self):
        self.model = None
        self.best_f1 = None
        self.best_auc = None

        try:
            test_model = XGBClassifier(tree_method='hist', device='cuda')
            test_model.set_params(n_estimators=1)
            test_model.fit(np.array([[0], [1]]), [0, 1])
            self.use_gpu = True
        except:
            self.use_gpu = False

    def fit(self, X_train_transformed, y_train, X_test_transformed, y_test):
        if isinstance(X_train_transformed, pd.DataFrame):
            feature_names = X_train_transformed.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]
            X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)

        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'random_state': 42,
                'verbosity': 0,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'tree_method': 'hist',
                'device': 'cuda' if self.use_gpu else 'cpu'
            }

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_scores = []
            auc_scores = []

            for train_idx, val_idx in skf.split(X_train_transformed, y_train):
                X_train_fold = X_train_transformed.iloc[train_idx]
                X_val_fold = X_train_transformed.iloc[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]

                model = XGBClassifier(**params)
                model.fit(X_train_fold, y_train_fold)

                preds = model.predict(X_val_fold)
                probas = model.predict_proba(X_val_fold)[:, 1]

                f1 = f1_score(y_val_fold, preds, average='macro')
                auc = roc_auc_score(y_val_fold, probas)

                f1_scores.append(f1)
                auc_scores.append(auc)

            trial.set_user_attr("auc_mean", np.mean(auc_scores))
            return np.mean(auc_scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        best_trial = study.best_trial
        self.best_f1 = best_trial.value
        self.best_auc = best_trial.user_attrs["auc_mean"]

        final_params = {
            **study.best_params,
            'random_state': 42,
            'use_label_encoder': False,
            'verbosity': 0,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu'
        }

        best_model = XGBClassifier(**final_params)

        experiment_name = "XGB trainer"
        try:
            mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            pass
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            best_model.fit(X_train_transformed, y_train)
            probas = best_model.predict_proba(X_test_transformed)[:, 1]
            auc = roc_auc_score(y_test, probas)
            f1_score_value = f1_score(y_test, best_model.predict(X_test_transformed), average='macro')
            mlflow.log_param("model_type", "XGB")
            mlflow.log_params(final_params)
            mlflow.log_metric("XGB_f1", f1_score_value)
            mlflow.log_metric("XGB_auc", auc)
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="XGB_Trainer")
            run_id = run.info.run_id
            mlflow.sklearn.save_model(best_model, f'/home/minhle/fastapi_model_serving/model/XGB/{run_id}')

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open("last_xgb_run_id.txt", "a") as f:
                f.write(f"{timestamp} - {run_id}\n")

        print(f"✅ Finished MLflow run: {run_id}")
        return {'run_id': run_id}
