import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
import optuna

class XGB_Trainer:
    def __init__(self):
        self.model = None
        self.best_f1 = None
        self.best_auc = None

        # Kiểm tra có GPU khả dụng không
        try:
            test_model = XGBClassifier(tree_method='hist', device='cuda')
            test_model.set_params(n_estimators=1)
            test_model.fit(np.array([[0], [1]]), [0, 1])
            self.use_gpu = True
        except:
            self.use_gpu = False


    def fit(self, X_train_transformed, y_train):
        # Đặt tên cột nếu cần
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
            return np.mean(f1_scores)

        # Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        best_trial = study.best_trial
        self.best_f1 = best_trial.value
        self.best_auc = best_trial.user_attrs["auc_mean"]

        # Thêm lại GPU params nếu cần
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
        best_model.fit(X_train_transformed, y_train)

        self.model = best_model
        return best_model, self.best_auc
