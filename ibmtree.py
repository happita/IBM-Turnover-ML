import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree 

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

    y = y.map({0:0, 1:1, 2:1}) 
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
    X, y, test_size=0.2, random_state=85, stratify=y
)

smote = SMOTE(random_state=85)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", np.bincount(y_resampled))


model = DecisionTreeClassifier(random_state=2, max_depth=5)
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # احتمال برای کلاس 1 (ترک خدمت)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score: {roc_auc:.2f}")

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

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob, pos_label=1)

desired_recall = 0.74 

indices = np.where(recall >= desired_recall)[0]

if len(indices) > 0:
    optimal_idx = indices[-1]
    optimal_threshold = thresholds_pr[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    print(f"\nOptimal Threshold: {optimal_threshold:.2f}")
    print(f"Precision at Optimal Threshold: {optimal_precision:.2f}")
    print(f"Recall at Optimal Threshold: {optimal_recall:.2f}")
else:
    print("\nNo threshold found that meets the desired recall.")
    optimal_threshold = 0.5  

y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f"\nAccuracy (Optimal Threshold): {accuracy_optimal:.2f}")
print("\nClassification Report (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal))
print("\nConfusion Matrix (Optimal Threshold):")
print(confusion_matrix(y_test, y_pred_optimal))

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
plt.scatter(optimal_recall, optimal_precision, color='red', label='Optimal Threshold')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], fontsize=10)
plt.title("Decision Tree Structure")
plt.show()
