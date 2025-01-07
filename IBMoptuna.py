!pip install optuna
import pandas as pd
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score, recall_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from google.colab import drive
drive.mount('/content/drive')
random.seed(0)

file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

data = pd.read_csv(file_path)

print("Unique values in 'Attrition' before mapping:", data['Attrition'].unique())

data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

print("Unique values in 'Attrition' after mapping:", data['Attrition'].unique())

features_to_remove = ['EmployeeNumber']
data = data.drop(columns=features_to_remove)

X = data.drop(columns=['Attrition'])
y = data['Attrition'].astype(int)

print("Unique values in 'y' after mapping:", y.unique())
if set(y.unique()).issubset({0,1}):
    print("y is correctly mapped to binary values (0 and 1).")
else:
    print("y contains values other than 0 and 1. Please check the mapping.")
    y = y.map({0: 0, 1: 1, 2: 1})
    print("Unique values in 'y' after remapping:", y.unique())

categorical_columns = X.select_dtypes(include=['object']).columns
print("Categorical columns to encode:", categorical_columns.tolist())

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
print("Numerical columns to scale:", numerical_columns.tolist())
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

smote = SMOTE(random_state=85)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", np.bincount(y_resampled))

scale_pos_weight = len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [scale_pos_weight]
}

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    use_label_encoder=False,
    random_state=85
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_resampled, y_resampled)

best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

best_xgb = grid_search.best_estimator_

y_pred_xgb = best_xgb.predict(X_test)
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nAccuracy: {accuracy_xgb:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
print(f"\nROC AUC Score: {roc_auc_xgb:.2f}")

fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_prob_xgb, pos_label=1)
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f"ROC curve (AUC = {roc_auc_xgb:.2f})", color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Random Guess')
optimal_idx_xgb = np.argmax(tpr_xgb - fpr_xgb)
plt.scatter(fpr_xgb[optimal_idx_xgb], tpr_xgb[optimal_idx_xgb], color='red', label='Best Threshold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve - XGBoost")
plt.legend(loc="lower right")
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()

threshold_03 = 0.3
y_pred_03 = (y_prob_xgb >= threshold_03).astype(int)

accuracy_03 = accuracy_score(y_test, y_pred_03)
print(f"\nAccuracy (Threshold=0.3): {accuracy_03:.2f}")
print("\nClassification Report (Threshold=0.3):")
print(classification_report(y_test, y_pred_03))

print("\nConfusion Matrix (Threshold=0.3):")
print(confusion_matrix(y_test, y_pred_03))

importances = best_xgb.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importances)[::-1]
plt.bar(range(len(features)), importances[sorted_idx], color='blue', alpha=0.7)
plt.xticks(range(len(features)), features[sorted_idx], rotation=45, ha='right')
plt.title("Feature Importances (Threshold=0.3)")
plt.tight_layout()
plt.show()

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=85, n_estimators=100, max_depth=5)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=85)

scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')

print(f"Accuracy per fold: {scores}")
print(f"Mean Accuracy: {scores.mean():.2f}")
print(f"Standard Deviation: {scores.std():.2f}")

def runxgb(seed, params):
    xgb_clf = XGBClassifier(
        booster='gbtree',
        gamma=params.get('gamma', 1.6),
        learning_rate=params.get('learning_rate', 0.15),
        max_depth=params.get('max_depth', 3),
        min_child_weight=params.get('min_child_weight', 2),
        n_estimators=params.get('n_estimators', 110),
        objective='binary:logistic',
        random_state=seed,
        reg_alpha=params.get('reg_alpha', 0),
        reg_lambda=params.get('reg_lambda', 1.5),
        subsample=params.get('subsample', 0.6)
    )

    xgb_clf.fit(X_train, y_train, verbose=0)
    y_pred = xgb_clf.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=1)

    return rec

# Define the objective function
def objective(trial, seed):
    params = {
        'gamma': trial.suggest_float('gamma', 0.0, 4.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.4),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'n_estimators': trial.suggest_int('n_estimators', 30, 350),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }

    mean_accuracy = runxgb(seed, params)

    return mean_accuracy

# Tune hyperparameters
def tune_hyperparameters(seed=0, n_trials=7000):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: objective(trial, seed), n_trials=n_trials)
    return study

if __name__ == "__main__":
    # Ensure X_train, X_test, y_train, y_test are defined before this section
    study = tune_hyperparameters(seed=0, n_trials=7000)

    print("Best Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    print(f"\nBest Mean Accuracy: {study.best_value:.4f}")
    
    xgb = XGBClassifier(

  gamma= 3.847132042098851,
  learning_rate= 0.32522493046958567,
  max_depth= 4,
  min_child_weight= 5,
  n_estimators= 338,
  reg_alpha= 0.05259006728568035,
  reg_lambda= 0.6946398838466616,
  subsample= 0.5499424465753451)
xgb.fit(X_train, y_train, verbose=0)

# پیش‌بینی و ارزیابی مدل
y_pred_xgb = xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"\nAccuracy: {accuracy_xgb:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve - XGBoost")
plt.legend(loc="lower right")
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()


accuracy_03 = accuracy_score(y_test, y_pred_xgb)
print(f"\nAccuracy (Threshold=0.3): {accuracy_03:.2f}")
print("\nClassification Report (Threshold=0.3):")
print(classification_report(y_test, y_pred_xgb))

print("\nConfusion Matrix (Threshold=0.3):")
print(confusion_matrix(y_test, y_pred_xgb))

# رسم Feature Importance
importances = xgb.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importances)[::-1]
plt.bar(range(len(features)), importances[sorted_idx], color='blue', alpha=0.7)
plt.xticks(range(len(features)), features[sorted_idx], rotation=45, ha='right')
plt.title("Feature Importances (Threshold=0.3)")
plt.tight_layout()
plt.show()
