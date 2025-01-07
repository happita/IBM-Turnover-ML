import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# بارگذاری داده‌ها
file_path = '/Users/hamidrezatofighian/Documents/datapy/HR_DataSet.csv'
data = pd.read_csv(file_path)

# تبدیل ستون‌های متنی به عددی
data['Department'] = data['Department'].astype('category').cat.codes
data['salary'] = data['salary'].astype('category').cat.codes
# print(data)
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
y = data['left']

data = data.drop(columns=['left'])
# data = data.drop(columns=['satisfaction_level'])
# data = data.drop(columns=['Department'])
# data = data.drop(columns=['salary'])
# data = data.drop(columns=['last_evaluation'])
# data = data.drop(columns=['number_project'])
# data = data.drop(columns=['average_montly_hours'])
# data = data.drop(columns=['time_spend_company'])
# data = data.drop(columns=['promotion_last_5years'])
# data = data.drop(columns=['Work_accident'])

X=data
print(X)
# تقسیم داده‌ها به مجموعه آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)
# print(X_train)
# تعریف و آموزش مدل جنگل تصادفی
rf_model = RandomForestClassifier(n_estimators=100, random_state=69)
rf_model.fit(X_train, y_train)

# محاسبه Feature Importance
feature_importances = rf_model.feature_importances_
features = X.columns

# رسم نمودار Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue', edgecolor='black')
plt.xlabel("اهمیت ویژگی‌ها")
plt.title("Feature Importance در مدل جنگل تصادفی")
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show()

# گزارش نهایی مدل
y_pred = rf_model.predict(X_test)
print("ماتریس درهم‌ریختگی:")
print(confusion_matrix(y_test, y_pred))
print("\nگزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred))