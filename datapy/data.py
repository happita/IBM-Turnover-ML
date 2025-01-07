import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# بارگذاری داده‌ها
file_path = '/Users/hamidrezatofighian/Documents/datapy/HR_DataSet.csv'
data = pd.read_csv(file_path)

# تبدیل ستون‌های متنی به عددی
data['Department'] = data['Department'].astype('category').cat.codes
data['salary'] = data['salary'].astype('category').cat.codes

# حذف مقادیر پرت
columns_to_check_outliers = ['average_montly_hours', 'time_spend_company']
for column in columns_to_check_outliers:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# جدا کردن ویژگی‌ها و متغیر هدف
X = data.drop(columns=['left'])
y = data['left']

# تقسیم داده‌ها به مجموعه آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تعریف و آموزش مدل رگرسیون لجستیک
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# پیش‌بینی و ارزیابی مدل
y_pred = model.predict(X_test_scaled)
print("ماتریس درهم‌ریختگی:")
print(confusion_matrix(y_test, y_pred))
print("\nگزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred))

# رسم نمودار ROC
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('نمودار ROC برای رگرسیون لجستیک')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()

# محاسبه و رسم نمودار Feature Importance
coefficients = np.abs(model.coef_[0])
features = X.columns

# ایجاد DataFrame برای نمایش اهمیت ویژگی‌ها
feature_importance = pd.DataFrame({'Feature': features, 'Importance': coefficients})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# رسم نمودار میله‌ای برای اهمیت ویژگی‌ها
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='light:blue_r')
plt.title('اهمیت ویژگی‌ها در رگرسیون لجستیک')
plt.xlabel('مقدار اهمیت ویژگی‌ها (قدر مطلق ضرایب)')
plt.ylabel('ویژگی‌ها')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
