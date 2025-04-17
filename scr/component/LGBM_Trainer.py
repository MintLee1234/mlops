from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
import optuna
import pandas as pd
import numpy as np


class LGBM_Trainer:
    def __init__(self):
        self.model = None
        self.best_f1 = None
        self.best_auc = None

    def fit(self, X_train_transformed, y_train):
        if isinstance(X_train_transformed, pd.DataFrame):
            feature_names = X_train_transformed.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]
            X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)

        def objective(trial):
            params = {
                'random_state': 42,
                'verbose': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)
            }

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_scores = []
            auc_scores = []

            for train_idx, val_idx in skf.split(X_train_transformed, y_train):
                X_train_fold = X_train_transformed.iloc[train_idx]
                X_val_fold = X_train_transformed.iloc[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]

                model = LGBMClassifier(**params)
                model.fit(X_train_fold, y_train_fold)

                preds = model.predict(X_val_fold)
                probas = model.predict_proba(X_val_fold)[:, 1]

                f1 = f1_score(y_val_fold, preds, average='macro')
                f1_scores.append(f1)

                # AUC tính trên từng lớp, cần nhị phân hóa y nếu là bài toán đa lớp
                auc = roc_auc_score(y_val_fold, probas)
                auc_scores.append(auc)


            # Đăng thêm thông tin AUC vào study
            trial.set_user_attr("mean_f1", np.mean(f1_scores))
            trial.set_user_attr("mean_auc", np.mean(auc_scores) if auc_scores else None)

            return np.mean(f1_scores)  # Optuna vẫn tối ưu theo F1

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        # Lưu lại thông tin F1 và AUC tốt nhất
        self.best_f1 = study.best_value
        self.best_auc = study.best_trial.user_attrs.get("mean_auc")

        best_model = LGBMClassifier(
            **study.best_params,
            random_state=42,
            verbose=-1
        )
        best_model.fit(X_train_transformed, y_train)

        self.model = best_model
        return best_model, self.best_auc
