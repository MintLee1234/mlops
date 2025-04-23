import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize



class ModelEvaluation:
    def __init__(self, model):
        self.model = model
        

    def predictions(self, X_val_transformed):
        y_pred = self.model.predict(X_val_transformed)
        y_pred_proba = self.model.predict_proba(X_val_transformed)  # For ROC-AUC and PR-AUC
        return y_pred, y_pred_proba

    def model_evaluation(self, y_val, y_pred, y_pred_proba):
        # Evaluation metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_val, y_pred_proba[:, 1])
        pr_auc = average_precision_score(y_val, y_pred_proba[:, 1])




        # Print detailed classification report and confusion matrix
        print("Classification Report:\n", classification_report(y_val, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

        
        # Return evaluation metrics
        return accuracy, precision, recall, f1, roc_auc, pr_auc

    def plot_roc_curve(self, y_val, y_pred_proba):
        # Lấy xác suất class 1
        y_score = y_pred_proba[:, 1]

        # Tính FPR, TPR và AUC
        fpr, tpr, _ = roc_curve(y_val, y_score)
        roc_auc = auc(fpr, tpr)

        # Vẽ biểu đồ
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_pr_curve(self, y_val, y_pred_proba):
        # Lấy xác suất class 1
        y_score = y_pred_proba[:, 1]

        # Tính Precision, Recall và AUC
        precision, recall, _ = precision_recall_curve(y_val, y_score)
        pr_auc = auc(recall, precision)

        # Vẽ biểu đồ
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        positive_rate = y_val.mean()
        plt.hlines(positive_rate, 0, 1, colors='gray', linestyles='--', label='Baseline')
        plt.legend(loc="lower left")
        plt.show()
    