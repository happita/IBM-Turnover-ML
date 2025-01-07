import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# مسیر فایل داده‌ها
file_path = "/Users/hamidrezatofighian/Documents/IBM.csv"

# بارگذاری داده‌ها
data = pd.read_csv(file_path)

# تبدیل مقادیر Attrition به 0 و 1 و تبدیل به عدد صحیح
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)

# حذف ویژگی‌های EmployeeNumber و BusinessTravel
features_to_remove = ['EmployeeNumber']
data = data.drop(columns=features_to_remove)

# تبدیل متغیرهای متنی به اعداد
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# مشاهده خلاصه آماری داده‌ها
print("Descriptive Statistics after Removing Outliers:")
print(data.describe())

# # رسم هیستوگرام برای متغیرهای عددی
# numerical_columns = ['Age', 'YearsWithCurrManager', 'YearsSinceLastPromotion', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']
# for column in numerical_columns:
#     plt.figure(figsize=(8, 4))
#     plt.hist(data[column], bins=20, edgecolor='k', alpha=0.7)
#     plt.title(f"Histogram of {column}")
#     plt.xlabel(column)
#     plt.ylabel("Frequency")
#     plt.show()


# # رسم نمودار پای برای متغیر Attrition
# attrition_counts = data['Attrition'].value_counts()
# plt.figure(figsize=(6, 6))
# plt.pie(attrition_counts, labels=['Stayed', 'Left'], autopct='%1.1f%%', colors=['lightblue', 'salmon'], startangle=90)
# plt.title("Attrition Distribution")
# plt.show()
# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

# # نمودار دایره‌ای برای جنسیت
# gender_counts = data['Gender'].value_counts()
# plt.figure(figsize=(6, 6))
# plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', colors=['skyblue', 'pink'], startangle=90)
# plt.title("Gender Distribution")
# plt.show()

# # نمودار دایره‌ای برای وضعیت تأهل
# marital_status_counts = data['MaritalStatus'].value_counts()
# plt.figure(figsize=(6, 6))
# plt.pie(marital_status_counts, labels=marital_status_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'salmon', 'gold'], startangle=90)
# plt.title("Marital Status Distribution")
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# # تنظیمات برای نمایش کامل ماتریس در کنسول
# import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# محاسبه ماتریس همبستگی
correlation_matrix = data.corr()

# # چاپ ماتریس همبستگی به صورت کامل
# print("Correlation Matrix:")
# print(correlation_matrix)

# # رسم هیت‌مپ با تنظیمات کامل و فاصله مناسب
# plt.figure(figsize=(20, 18))
# sns.heatmap(
#     correlation_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     cbar=True,
#     vmin=-1,
#     vmax=1,
#     linewidths=0.5,
#     annot_kws={"size": 8}  # تنظیم اندازه متن اعداد
# )
# plt.xticks(rotation=45, fontsize=10)
# plt.yticks(fontsize=10)
# plt.title("Correlation Matrix Heatmap", fontsize=14)
# plt.tight_layout()
# plt.show()

# استانداردسازی داده‌ها
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# انتخاب ویژگی‌ها و هدف
X = data.drop(columns=['Attrition'])
y = data['Attrition'].astype(int)  # تبدیل به عدد صحیح

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=85, stratify=y
)

# استفاده از SMOTE برای افزایش نمونه‌های کلاس اقلیت
smote = SMOTE(random_state=85)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ایجاد و آموزش مدل رگرسیون لجستیک
model = LogisticRegression(max_iter=1000, random_state=85)
model.fit(X_resampled, y_resampled)

# پیش‌بینی روی داده‌های تست
y_pred = model.predict(X_test)

# ارزیابی مدل
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# تبدیل مقادیر y_test از {0, 2} به {0, 1}
y_test_binary = y_test.replace(2, 1)

# پیش‌بینی احتمال‌ها برای داده‌های تست
y_prob = model.predict_proba(X_test)[:, 1]  # احتمال برای کلاس 1 (ترک خدمت)

# محاسبه مقادیر FPR، TPR و Thresholds
fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob)

# محاسبه AUC
roc_auc = roc_auc_score(y_test_binary, y_prob)
print(f"ROC AUC Score: {roc_auc:.2f}")

# رسم منحنی ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()