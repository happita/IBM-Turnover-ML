# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt

# # مسیر فایل داده‌ها
# file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

# # بارگذاری داده‌ها
# data = pd.read_csv(file_path)

# # تبدیل مقادیر Attrition به 0 و 1 و تبدیل به عدد صحیح
# data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

# # حذف ویژگی‌های EmployeeNumber
# features_to_remove = ['EmployeeNumber']
# data = data.drop(columns=features_to_remove)

# # تبدیل متغیرهای متنی به اعداد
# categorical_columns = data.select_dtypes(include=['object']).columns
# label_encoders = {}
# for column in categorical_columns:
#     le = LabelEncoder()
#     data[column] = le.fit_transform(data[column])
#     label_encoders[column] = le

# # انتخاب ویژگی‌ها و هدف
# X = data.drop(columns=['Attrition'])
# y = data['Attrition'].astype(int)  # تبدیل به عدد صحیح

# # تقسیم داده‌ها به آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1, stratify=y
# )

# # استفاده از SMOTE برای افزایش نمونه‌های کلاس اقلیت
# smote = SMOTE(random_state=85)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# # محاسبه نسبت کلاس‌ها برای تنظیم scale_pos_weight
# scale_pos_weight = len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])

# # ایجاد و آموزش مدل XGBoost با تنظیم scale_pos_weight
# model = XGBClassifier(
#     n_estimators=300,               # افزایش تعداد درخت‌ها
#     max_depth=20,                    # حداکثر عمق درخت‌ها
#     learning_rate=0.1,             # کاهش نرخ یادگیری
#     random_state=1,                # ثابت نگه‌داشتن تصادفی بودن
#     # scale_pos_weight=scale_pos_weight,  # وزن‌دهی کلاس‌ها
#     use_label_encoder=False,
#     eval_metric='aucpr'             # استفاده از متریک aucpr
# )
# model.fit(X_resampled, y_resampled)

# # پیش‌بینی روی داده‌های تست
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]  # احتمال پیش‌بینی برای کلاس مثبت

# # ارزیابی مدل
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nAccuracy: {accuracy:.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # رسم منحنی ROC
# fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")
# plt.title("ROC Curve - XGBoost")
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)
# plt.show()




# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from xgboost import XGBClassifier, plot_tree
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     roc_auc_score, roc_curve, precision_recall_curve, f1_score
# )
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # مسیر فایل داده‌ها
# file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

# # بارگذاری داده‌ها
# data = pd.read_csv(file_path)

# # بررسی مقادیر یکتا در ستون Attrition قبل از تبدیل
# print("Unique values in 'Attrition' before mapping:", data['Attrition'].unique())

# # تبدیل مقادیر Attrition به 0 و 1 با استفاده از نگاشت دقیق
# data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

# # بررسی مقادیر یکتا در ستون Attrition پس از تبدیل
# print("Unique values in 'Attrition' after mapping:", data['Attrition'].unique())

# # حذف ویژگی‌های EmployeeNumber
# features_to_remove = ['EmployeeNumber']
# data = data.drop(columns=features_to_remove)

# # جدا کردن هدف
# X = data.drop(columns=['Attrition'])
# y = data['Attrition'].astype(int)  # اطمینان از تبدیل به عدد صحیح

# # بررسی مقادیر یکتا در y بعد از تبدیل
# print("Unique values in 'y' after mapping:", y.unique())

# # اطمینان از اینکه 'y' تنها شامل 0 و 1 باشد
# if set(y.unique()).issubset({0,1}):
#     print("y is correctly mapped to binary values (0 and 1).")
# else:
#     print("y contains values other than 0 and 1. Please check the mapping.")
#     # در اینجا می‌توانید مقادیر اضافی را به 1 نگاشت کنید، اگر منطقی است
#     y = y.map({0:0, 1:1, 2:1})  # به عنوان مثال، مقادیر 2 را به 1 نگاشت می‌کنیم
#     print("Unique values in 'y' after remapping:", y.unique())

# # تبدیل متغیرهای متنی به اعداد، بدون 'Attrition' چون اکنون عددی است
# categorical_columns = X.select_dtypes(include=['object']).columns
# print("Categorical columns to encode:", categorical_columns.tolist())

# label_encoders = {}
# for column in categorical_columns:
#     le = LabelEncoder()
#     X[column] = le.fit_transform(X[column])
#     label_encoders[column] = le

# # استانداردسازی داده‌ها (اختیاری برای مدل‌های درختی مانند XGBoost ضروری نیست، اما برای برخی داده‌ها مفید است)
# numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
# print("Numerical columns to scale:", numerical_columns.tolist())

# scaler = StandardScaler()
# X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# # تقسیم داده‌ها به آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1, stratify=y
# )

# # استفاده از SMOTE برای افزایش نمونه‌های کلاس اقلیت
# smote = SMOTE(random_state=85)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# # بررسی توزیع کلاس‌ها پس از SMOTE
# print("Class distribution after SMOTE:", np.bincount(y_resampled))

# # محاسبه نسبت کلاس‌ها برای تنظیم scale_pos_weight
# scale_pos_weight = len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])
# print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

# # ============================================
# # ۱. تنظیم هایپرپارامترها با GridSearchCV
# # ============================================

# # تعریف فضای جستجو برای پارامترها
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'gamma': [0, 0.1, 0.2],
#     'min_child_weight': [1, 3, 5],
#     'scale_pos_weight': [scale_pos_weight]  # استفاده از مقدار محاسبه شده
# }

# # ایجاد مدل XGBoost پایه
# xgb = XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='aucpr',
#     use_label_encoder=False,
#     random_state=85
# )

# # ایجاد GridSearchCV
# grid_search = GridSearchCV(
#     estimator=xgb,
#     param_grid=param_grid,
#     scoring='f1',  # می‌توانید از معیارهای دیگر مانند 'roc_auc' استفاده کنید
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )

# # اجرای GridSearchCV
# grid_search.fit(X_resampled, y_resampled)

# # بهترین پارامترها
# best_params = grid_search.best_params_
# print(f"Best Parameters: {best_params}")

# # ایجاد مدل با بهترین پارامترها
# best_xgb = grid_search.best_estimator_

# # ============================================
# # ۲. ارزیابی مدل XGBoost
# # ============================================

# # پیش‌بینی روی داده‌های تست
# y_pred_xgb = best_xgb.predict(X_test)
# y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]  # احتمال برای کلاس 1 (ترک خدمت)

# # محاسبه دقت
# accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
# print(f"\nAccuracy: {accuracy_xgb:.2f}")

# # محاسبه گزارش طبقه‌بندی
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_xgb))

# # محاسبه ماتریس درهم‌ریختگی
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred_xgb))

# # محاسبه ROC AUC
# roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)
# print(f"\nROC AUC Score: {roc_auc_xgb:.2f}")

# # رسم منحنی ROC
# fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_prob_xgb, pos_label=1)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_xgb, tpr_xgb, label=f"ROC curve (AUC = {roc_auc_xgb:.2f})", color='orange')
# plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Random Guess')
# optimal_idx_xgb = np.argmax(tpr_xgb - fpr_xgb)
# plt.scatter(fpr_xgb[optimal_idx_xgb], tpr_xgb[optimal_idx_xgb], color='red', label='Best Threshold')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC) Curve - XGBoost")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.grid(alpha=0.3)
# plt.show()

# # ============================================
# # ۳. تنظیم آستانه بهینه (Threshold Tuning)
# # ============================================

# # محاسبه منحنی Precision-Recall
# precision_xgb, recall_xgb, thresholds_pr_xgb = precision_recall_curve(y_test, y_prob_xgb, pos_label=1)

# # تعیین آستانه بهینه بر اساس مقدار Recall مورد نظر
# desired_recall_xgb = 0.74  # مقدار یادآوری مورد نظر

# # پیدا کردن ایندکس‌هایی که Recall بیشتر یا مساوی به مقدار مورد نظر دارند
# indices_xgb = np.where(recall_xgb >= desired_recall_xgb)[0]

# if len(indices_xgb) > 0:
#     optimal_idx_xgb = indices_xgb[-1]  # آخرین ایندکس که شرط را دارد
#     optimal_threshold_xgb = thresholds_pr_xgb[optimal_idx_xgb]
#     optimal_precision_xgb = precision_xgb[optimal_idx_xgb]
#     optimal_recall_xgb = recall_xgb[optimal_idx_xgb]
#     print(f"\nOptimal Threshold: {optimal_threshold_xgb:.2f}")
#     print(f"Precision at Optimal Threshold: {optimal_precision_xgb:.2f}")
#     print(f"Recall at Optimal Threshold: {optimal_recall_xgb:.2f}")
# else:
#     print("\nNo threshold found that meets the desired recall.")
#     optimal_threshold_xgb = 0.5  # آستانه پیش‌فرض

# # پیش‌بینی با آستانه بهینه
# y_pred_optimal_xgb = (y_prob_xgb >= optimal_threshold_xgb).astype(int)

# # ارزیابی مدل با آستانه بهینه
# accuracy_optimal_xgb = accuracy_score(y_test, y_pred_optimal_xgb)
# print(f"\nAccuracy (Optimal Threshold): {accuracy_optimal_xgb:.2f}")
# print("\nClassification Report (Optimal Threshold):")
# print(classification_report(y_test, y_pred_optimal_xgb))
# print("\nConfusion Matrix (Optimal Threshold):")
# print(confusion_matrix(y_test, y_pred_optimal_xgb))

# # رسم منحنی Precision-Recall با آستانه بهینه
# plt.figure(figsize=(8, 6))
# plt.plot(recall_xgb, precision_xgb, color='green', lw=2, label='Precision-Recall Curve')
# plt.scatter(optimal_recall_xgb, optimal_precision_xgb, color='red', label='Optimal Threshold')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve - XGBoost')
# plt.legend(loc='upper right')
# plt.grid(alpha=0.3)
# plt.show()

# # ============================================
# # ۴. رسم نمودار ساختار درخت تصمیم (رفع خطا)
# # ============================================

# # توجه: در جنگل تصادفی و XGBoost، مدل شامل چندین درخت است. برای رسم یک درخت مشخص، می‌توانید یک درخت انتخاب کنید.

# # انتخاب یک درخت از مدل XGBoost برای رسم
# plt.figure(figsize=(20,10))
# plot_tree(best_xgb, num_trees=0, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, fontsize=10)
# plt.title("XGBoost Decision Tree Structure")
# plt.show()




import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# مسیر فایل داده‌ها
file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

# بارگذاری داده‌ها
data = pd.read_csv(file_path)

# بررسی مقادیر یکتا در ستون Attrition قبل از تبدیل
print("Unique values in 'Attrition' before mapping:", data['Attrition'].unique())

# تبدیل مقادیر Attrition به 0 و 1 با استفاده از نگاشت دقیق
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

# بررسی مقادیر یکتا در ستون Attrition پس از تبدیل
print("Unique values in 'Attrition' after mapping:", data['Attrition'].unique())

# حذف ویژگی‌های EmployeeNumber
features_to_remove = ['EmployeeNumber']
data = data.drop(columns=features_to_remove)

# جدا کردن هدف
X = data.drop(columns=['Attrition'])
y = data['Attrition'].astype(int)  # اطمینان از تبدیل به عدد صحیح

# بررسی مقادیر یکتا در y بعد از تبدیل
print("Unique values in 'y' after mapping:", y.unique())
if set(y.unique()).issubset({0,1}):
    print("y is correctly mapped to binary values (0 and 1).")
else:
    print("y contains values other than 0 and 1. Please check the mapping.")
    y = y.map({0: 0, 1: 1, 2: 1})
    print("Unique values in 'y' after remapping:", y.unique())

# تبدیل متغیرهای متنی به اعداد
categorical_columns = X.select_dtypes(include=['object']).columns
print("Categorical columns to encode:", categorical_columns.tolist())

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# استانداردسازی داده‌ها (اختیاری برای XGBoost، اما گاهی مفید)
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
print("Numerical columns to scale:", numerical_columns.tolist())
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# اعمال SMOTE برای افزایش نمونه‌های کلاس اقلیت
smote = SMOTE(random_state=85)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# بررسی توزیع کلاس‌ها پس از SMOTE
print("Class distribution after SMOTE:", np.bincount(y_resampled))

# محاسبه نسبت کلاس‌ها برای تنظیم scale_pos_weight
scale_pos_weight = len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

# تعریف فضای جستجو برای پارامترها
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

# پیش‌بینی و ارزیابی مدل
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

# تنظیم آستانه 0.3
threshold_03 = 0.3
y_pred_03 = (y_prob_xgb >= threshold_03).astype(int)

accuracy_03 = accuracy_score(y_test, y_pred_03)
print(f"\nAccuracy (Threshold=0.3): {accuracy_03:.2f}")
print("\nClassification Report (Threshold=0.3):")
print(classification_report(y_test, y_pred_03))

print("\nConfusion Matrix (Threshold=0.3):")
print(confusion_matrix(y_test, y_pred_03))

# رسم Feature Importance
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

# تنظیم مدل
xgb_model = XGBClassifier(random_state=85, n_estimators=100, max_depth=5)

# تنظیم 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=85)

# محاسبه دقت
scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')

# گزارش نتایج
print(f"Accuracy per fold: {scores}")
print(f"Mean Accuracy: {scores.mean():.2f}")
print(f"Standard Deviation: {scores.std():.2f}")