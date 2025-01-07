import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import tree
from graphviz import Source

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
X = data.drop(columns=['left'])  # فرض بر این است که ستون 'left' هدف است
y = data['left']

# تقسیم داده‌ها به مجموعه آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تعریف مدل درخت تصمیم
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)  # تنظیم عمق درخت

# آموزش مدل
decision_tree.fit(X_train, y_train)

# پیش‌بینی روی داده‌های آزمایشی
y_pred = decision_tree.predict(X_test)
y_prob = decision_tree.predict_proba(X_test)[:, 1]  # احتمال پیش‌بینی برای کلاس 1

# ارزیابی مدل
print("ماتریس درهم‌ریختگی:")
print(confusion_matrix(y_test, y_pred))

print("\nگزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred))

# رسم درخت تصمیم با Matplotlib
plt.figure(figsize=(20, 10))
tree.plot_tree(
    decision_tree, 
    feature_names=X.columns, 
    class_names=['Not Left', 'Left'], 
    filled=True, 
    rounded=True
)
plt.title("نمودار درخت تصمیم")
plt.savefig("decision_tree_plot.png", dpi=300)  # ذخیره نمودار با کیفیت بالا
plt.show()

# رسم نمودار ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # خط تصادفی
plt.title("منحنی ROC درخت تصمیم")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("decision_tree_roc.png", dpi=300)  # ذخیره نمودار ROC با کیفیت بالا
plt.show()

# تولید فایل .dot برای Graphviz
dot_data = export_graphviz(
    decision_tree,  # مدل درخت تصمیم
    out_file=None,
    feature_names=X.columns,  # ویژگی‌ها
    class_names=['Not Left', 'Left'],  # نام کلاس‌ها
    filled=True,
    rounded=True,
    special_characters=True
)

# نمایش گراف با Graphviz
graph = Source(dot_data)
graph.render("decision_tree")  # ذخیره نمودار درخت به‌صورت فایل PDF
graph.view()  # مشاهده نمودار در یک پنجره جداگانه

# محاسبه اهمیت ویژگی‌ها
feature_importances = pd.Series(decision_tree.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')
plt.title("اهمیت ویژگی‌ها در مدل درخت تصمیم")
plt.ylabel("میزان اهمیت")
plt.xlabel("ویژگی‌ها")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("decision_tree_feature_importance.png", dpi=300)  # ذخیره نمودار اهمیت ویژگی‌ها
plt.show()
