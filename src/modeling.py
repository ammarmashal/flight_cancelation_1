# Model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic
import lightgbm as lgb
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(X_train, y_train, X_val=None, y_val=None):
    class_weights = compute_sample_weight('balanced', y_train)
    train_data = lgb.Dataset(X_train, label=y_train, weight=class_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data) if X_val is not None else None

    params = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'random_state': 42
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data] if val_data else [train_data],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    print("Classification Report:")
    print(classification_report(y, y_pred_binary))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred_binary))
    print(f"\nROC AUC: {roc_auc_score(y, y_pred):.4f}")
    print(f"PR AUC: {average_precision_score(y, y_pred):.4f}")
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=20)
    plt.title('Feature Importance')
    plt.show()
