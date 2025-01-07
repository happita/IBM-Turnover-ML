import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# بارگذاری داده‌ها
file_path = '/Users/hamidrezatofighian/Documents/datapy/HR_DataSet.csv'
data = pd.read_csv(file_path)

# بررسی داده‌های گمشده
print("تعداد داده‌های گمشده در هر ستون:")
print(data.isnull().sum())

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

# تعریف مدل با بهترین پارامترها
best_model = SVC(C=10, gamma=0.1, kernel='rbf', probability=True, random_state=42)

# آموزش مدل
best_model.fit(X_train_scaled, y_train)

# ذخیره مدل بهینه
joblib.dump(best_model, 'best_svm_model_rbf.pkl')

# پیش‌بینی و ارزیابی مدل
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# چاپ نتایج
print("ماتریس درهم‌ریختگی:")
print(confusion_matrix(y_test, y_pred))
print("\nگزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred))

# رسم نمودار ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('منحنی ROC برای SVM با کرنل RBF')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig('Optimized_SVM_RBF_ROC_Curve.png', dpi=300)
plt.show()