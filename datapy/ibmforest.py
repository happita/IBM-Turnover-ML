# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt

# file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

# data = pd.read_csv(file_path)

# data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

# features_to_remove = ['EmployeeNumber']
# data = data.drop(columns=features_to_remove)

# categorical_columns = data.select_dtypes(include=['object']).columns
# label_encoders = {}
# for column in categorical_columns:
#     le = LabelEncoder()
#     data[column] = le.fit_transform(data[column])
#     label_encoders[column] = le

# X = data.drop(columns=['Attrition'])
# y = data['Attrition'].astype(int)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=85, stratify=y
# )

# smote = SMOTE(random_state=85)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# model = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced', random_state=85)
# model.fit(X_resampled, y_resampled)

# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]  # احتمال پیش‌بینی برای کلاس مثبت

# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nAccuracy: {accuracy:.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")
# plt.title("ROC Curve - Random Forest")
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)
# plt.show()

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     roc_auc_score, roc_curve, precision_recall_curve
# )
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

# data = pd.read_csv(file_path)

# print("Unique values in 'Attrition' before mapping:", data['Attrition'].unique())

# data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

# print("Unique values in 'Attrition' after mapping:", data['Attrition'].unique())

# features_to_remove = ['EmployeeNumber']
# data = data.drop(columns=features_to_remove)

# X = data.drop(columns=['Attrition'])
# y = data['Attrition'].astype(int)  
# print("Unique values in 'y' after mapping:", y.unique())

# if set(y.unique()).issubset({0,1}):
#     print("y is correctly mapped to binary values (0 and 1).")
# else:
#     print("y contains values other than 0 and 1. Please check the mapping.")
#  
#     y = y.map({0:0, 1:1, 2:1}) 
#     print("Unique values in 'y' after remapping:", y.unique())


# categorical_columns = X.select_dtypes(include=['object']).columns
# print("Categorical columns to encode:", categorical_columns.tolist())

# label_encoders = {}
# for column in categorical_columns:
#     le = LabelEncoder()
#     X[column] = le.fit_transform(X[column])
#     label_encoders[column] = le


# numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
# print("Numerical columns to scale:", numerical_columns.tolist())

# scaler = StandardScaler()
# X[numerical_columns] = scaler.fit_transform(X[numerical_columns])


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=85, stratify=y
# )


# smote = SMOTE(random_state=85)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

#
# print("Class distribution after SMOTE:", np.bincount(y_resampled))



# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15],
#     'min_samples_split': [2, 5, 10],
#     'class_weight': ['balanced', 'balanced_subsample', None]
# }


# rf = RandomForestClassifier(random_state=85)


# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     scoring='f1',
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )


# grid_search.fit(X_resampled, y_resampled)


# best_params = grid_search.best_params_
# print(f"Best Parameters: {best_params}")


# best_rf = grid_search.best_estimator_



# y_pred_rf = best_rf.predict(X_test)
# y_prob_rf = best_rf.predict_proba(X_test)[:, 1]  # احتمال برای کلاس 1 (ترک خدمت)


# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"\nAccuracy: {accuracy_rf:.2f}")


# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_rf))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred_rf))

# roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
# print(f"\nROC AUC Score: {roc_auc_rf:.2f}")

# fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf, pos_label=1)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_rf, tpr_rf, label=f"ROC curve (AUC = {roc_auc_rf:.2f})", color='orange')
# plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Random Guess')
# optimal_idx_rf = np.argmax(tpr_rf - fpr_rf)
# plt.scatter(fpr_rf[optimal_idx_rf], tpr_rf[optimal_idx_rf], color='red', label='Best Threshold')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC) Curve - Random Forest")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.grid(alpha=0.3)
# plt.show()


# precision_rf, recall_rf, thresholds_pr_rf = precision_recall_curve(y_test, y_prob_rf, pos_label=1)


# desired_recall_rf = 0.74  


# indices_rf = np.where(recall_rf >= desired_recall_rf)[0]

# if len(indices_rf) > 0:
#     optimal_idx_rf = indices_rf[-1]  # آخرین ایندکس که شرط را دارد
#     optimal_threshold_rf = thresholds_pr_rf[optimal_idx_rf]
#     optimal_precision_rf = precision_rf[optimal_idx_rf]
#     optimal_recall_rf = recall_rf[optimal_idx_rf]
#     print(f"\nOptimal Threshold: {optimal_threshold_rf:.2f}")
#     print(f"Precision at Optimal Threshold: {optimal_precision_rf:.2f}")
#     print(f"Recall at Optimal Threshold: {optimal_recall_rf:.2f}")
# else:
#     print("\nNo threshold found that meets the desired recall.")
#     optimal_threshold_rf = 0.5  # آستانه پیش‌فرض


# y_pred_optimal_rf = (y_prob_rf >= optimal_threshold_rf).astype(int)


# accuracy_optimal_rf = accuracy_score(y_test, y_pred_optimal_rf)
# print(f"\nAccuracy (Optimal Threshold): {accuracy_optimal_rf:.2f}")
# print("\nClassification Report (Optimal Threshold):")
# print(classification_report(y_test, y_pred_optimal_rf))
# print("\nConfusion Matrix (Optimal Threshold):")
# print(confusion_matrix(y_test, y_pred_optimal_rf))

# plt.figure(figsize=(8, 6))
# plt.plot(recall_rf, precision_rf, color='green', lw=2, label='Precision-Recall Curve')
# plt.scatter(optimal_recall_rf, optimal_precision_rf, color='red', label='Optimal Threshold')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve - Random Forest')
# plt.legend(loc='upper right')
# plt.grid(alpha=0.3)
# plt.show()


# importances = best_rf.feature_importances_
# feature_names = X.columns
# feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# print("\nTop 10 Feature Importances:")
# print(feature_importance_df.head(10))

# plt.figure(figsize=(10, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
# plt.title('Top 10 Feature Importances - Random Forest')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=85)

# # ارزیابی مدل با استفاده از Cross-Validation
# cv_scores_rf = GridSearchCV(
#     estimator=RandomForestClassifier(**best_params, random_state=85),
#     param_grid={},
#     scoring='f1',
#     cv=skf,
#     n_jobs=-1,
#     verbose=0
# ).fit(X_resampled, y_resampled).best_score_

# print(f"\nCross-Validation F1 Score: {cv_scores_rf:.2f}")

# errors_rf = X_test[y_test != y_pred_rf]
# errors_rf = errors_rf.copy()
# errors_rf['Actual'] = y_test[y_test != y_pred_rf]
# errors_rf['Predicted'] = y_pred_rf[y_test != y_pred_rf]

# print("\nMisclassified Samples:")
# print(errors_rf.head())

# for index, row in errors_rf.head(5).iterrows():
#     print(f"\nSample Index: {index}")
#     print(row)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

data = pd.read_csv(file_path)

data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

features_to_remove = ['EmployeeNumber']
data = data.drop(columns=features_to_remove)

categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop(columns=['Attrition'])
y = data['Attrition'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=85, stratify=y
)

smote = SMOTE(random_state=85)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced', random_state=85)
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # احتمال پیش‌بینی برای کلاس مثبت

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
