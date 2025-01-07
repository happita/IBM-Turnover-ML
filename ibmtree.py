# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns

# # مسیر فایل داده‌ها
# file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

# # بارگذاری داده‌ها
# data = pd.read_csv(file_path)

# # تبدیل مقادیر Attrition به 0 و 1 و تبدیل به عدد صحیح
# data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

# # حذف ویژگی‌های EmployeeNumber و BusinessTravel
# features_to_remove = ['EmployeeNumber']
# data = data.drop(columns=features_to_remove)

# # تبدیل متغیرهای متنی به اعداد
# categorical_columns = data.select_dtypes(include=['object']).columns
# label_encoders = {}
# for column in categorical_columns:
#     le = LabelEncoder()
#     data[column] = le.fit_transform(data[column])
#     label_encoders[column] = le

# # استانداردسازی داده‌ها
# numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
# scaler = StandardScaler()
# data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# # انتخاب ویژگی‌ها و هدف
# X = data.drop(columns=['Attrition'])
# y = data['Attrition'].astype(int)

# # تقسیم داده‌ها به آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=85, stratify=y
# )

# # استفاده از SMOTE برای افزایش نمونه‌های کلاس اقلیت
# smote = SMOTE(random_state=2)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# # ایجاد و آموزش مدل درخت تصمیم
# model = DecisionTreeClassifier(random_state=2, max_depth=5)
# model.fit(X_resampled, y_resampled)

# # پیش‌بینی روی داده‌های تست
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]

# # ارزیابی مدل
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nAccuracy: {accuracy:.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # # رسم نمودار ROC
# # from sklearn.metrics import roc_curve, roc_auc_score

# # fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=2)  # اضافه کردن pos_label=2
# # roc_auc = roc_auc_score(y_test, y_prob)
# # plt.figure(figsize=(8, 6))
# # plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color='orange')
# # plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Random Guess')
# # plt.xlabel("False Positive Rate")
# # plt.ylabel("True Positive Rate")
# # plt.title("Receiver Operating Characteristic (ROC) Curve")
# # plt.legend(loc="lower right")
# # plt.tight_layout()
# # plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree  # اضافه کردن کتابخانه درخت

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

# اطمینان از اینکه 'y' تنها شامل 0 و 1 باشد
if set(y.unique()).issubset({0,1}):
    print("y is correctly mapped to binary values (0 and 1).")
else:
    print("y contains values other than 0 and 1. Please check the mapping.")
    # در اینجا می‌توانید مقادیر اضافی را به 1 نگاشت کنید، اگر منطقی است
    y = y.map({0:0, 1:1, 2:1})  # به عنوان مثال، مقادیر 2 را به 1 نگاشت می‌کنیم
    print("Unique values in 'y' after remapping:", y.unique())

# تبدیل متغیرهای متنی به اعداد، بدون 'Attrition' چون اکنون عددی است
categorical_columns = X.select_dtypes(include=['object']).columns
print("Categorical columns to encode:", categorical_columns.tolist())

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# استانداردسازی داده‌ها
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
print("Numerical columns to scale:", numerical_columns.tolist())

scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=85, stratify=y
)

# استفاده از SMOTE برای افزایش نمونه‌های کلاس اقلیت
smote = SMOTE(random_state=85)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# بررسی توزیع کلاس‌ها پس از SMOTE
print("Class distribution after SMOTE:", np.bincount(y_resampled))

# ============================================
# ۱. ایجاد و آموزش مدل درخت تصمیم
# ============================================

# ایجاد و آموزش مدل درخت تصمیم با حداکثر عمق 5
model = DecisionTreeClassifier(random_state=2, max_depth=5)
model.fit(X_resampled, y_resampled)

# پیش‌بینی روی داده‌های تست
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # احتمال برای کلاس 1 (ترک خدمت)

# ============================================
# ۲. ارزیابی مدل
# ============================================

# محاسبه دقت
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# محاسبه گزارش طبقه‌بندی
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# محاسبه ماتریس درهم‌ریختگی
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# محاسبه ROC AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score: {roc_auc:.2f}")

# رسم منحنی ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color='orange')
plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Random Guess')
plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], color='red', label='Best Threshold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()

# ============================================
# ۳. تنظیم آستانه بهینه (Threshold Tuning)
# ============================================

from sklearn.metrics import precision_recall_curve

# محاسبه منحنی Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob, pos_label=1)

# تعیین آستانه بهینه بر اساس مقدار Recall مورد نظر
desired_recall = 0.74  # مقدار یادآوری مورد نظر

# پیدا کردن ایندکس‌هایی که Recall بیشتر یا مساوی به مقدار مورد نظر دارند
indices = np.where(recall >= desired_recall)[0]

if len(indices) > 0:
    optimal_idx = indices[-1]  # آخرین ایندکس که شرط را دارد
    optimal_threshold = thresholds_pr[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    print(f"\nOptimal Threshold: {optimal_threshold:.2f}")
    print(f"Precision at Optimal Threshold: {optimal_precision:.2f}")
    print(f"Recall at Optimal Threshold: {optimal_recall:.2f}")
else:
    print("\nNo threshold found that meets the desired recall.")
    optimal_threshold = 0.5  # آستانه پیش‌فرض

# پیش‌بینی با آستانه بهینه
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

# ارزیابی مدل با آستانه بهینه
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f"\nAccuracy (Optimal Threshold): {accuracy_optimal:.2f}")
print("\nClassification Report (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal))
print("\nConfusion Matrix (Optimal Threshold):")
print(confusion_matrix(y_test, y_pred_optimal))

# رسم منحنی Precision-Recall با آستانه بهینه
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
plt.scatter(optimal_recall, optimal_precision, color='red', label='Optimal Threshold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.show()

# ============================================
# ۴. رسم نمودار ساختار درخت تصمیم
# ============================================

plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], fontsize=10)
plt.title("Decision Tree Structure")
plt.show()