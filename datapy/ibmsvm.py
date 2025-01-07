import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

# مسیر فایل داده‌ها
file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

# بارگذاری داده‌ها
data = pd.read_csv(file_path)

# --- تبدیل مقادیر Attrition به 0 و 1 و تبدیل به عدد صحیح (بدون تغییر) ---
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

# حذف ویژگی‌های غیرضروری
features_to_remove = ['EmployeeNumber']
data = data.drop(columns=features_to_remove)

# تبدیل متغیرهای متنی به اعداد
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# انتخاب ویژگی‌ها و هدف
X = data.drop(columns=['Attrition'])
y = data['Attrition']

# استانداردسازی داده‌ها
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=85, stratify=y
)

# اعمال SMOTE
smote = SMOTE(random_state=85)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# =======================
# ۱. بهبود مدل با استفاده از جستجوی شبکه‌ای (GridSearchCV) برای یافتن بهترین هایپرپارامترهای SVM
# =======================

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'class_weight': ['balanced', None]
}

svc = SVC(probability=True, random_state=85)

grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    scoring='accuracy',  # یا می‌توانید از 'precision' یا 'f1' استفاده کنید
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=85),
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_resampled, y_resampled)

best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

model = grid_search.best_estimator_

# =======================
# ۲. آموزش مدل نهایی و ارزیابی
# =======================

model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

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
plt.title("ROC Curve - SVM (Optimized)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# =======================
# ۳. تنظیم آستانه تصمیم‌گیری برای افزایش Precision
# =======================

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob, pos_label=1)
max_precision = precision.max()
max_precision_idx = precision.argmax()

if max_precision_idx < len(thresholds_pr):
    optimal_threshold = thresholds_pr[max_precision_idx]
else:
    optimal_threshold = 1.0

max_precision_recall = recall[max_precision_idx] if max_precision_idx < len(recall) else 0.0

print(f"\nThreshold for Maximum Precision: {optimal_threshold:.2f}")
print(f"Maximum Precision: {max_precision:.2f}")
print(f"Recall at Maximum Precision: {max_precision_recall:.2f}")

y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f"\nAccuracy (Optimal Threshold): {accuracy_optimal:.2f}")

print("\nClassification Report (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal))

print("\nConfusion Matrix (Optimal Threshold):")
print(confusion_matrix(y_test, y_pred_optimal))

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall Curve')
plt.scatter(max_precision_recall, max_precision, color='red', label='Optimal Threshold')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - SVM (Optimized)")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.show()