import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from knn import KNN  

file_path_cleaned = r"C:\Users\PureTech\Desktop\greymatter\youth destroyer\Literature\Article material\Algerian_forest_fires_dataset_CLEANED.csv"
df_cleaned = pd.read_csv(file_path_cleaned)

df_cleaned['Classes'] = df_cleaned['Classes'].str.strip().str.lower()
df_cleaned['Classes'] = df_cleaned['Classes'].replace({'fire': 'fire', 'not fire': 'not_fire'})
df_cleaned = df_cleaned[df_cleaned['Classes'].isin(['fire', 'not_fire'])]
print("Original class distribution:")
print(df_cleaned['Classes'].value_counts())

# ### **(A) Undersampling Majority Class (Reduce 'not_fire')**
# df_fire = df_cleaned[df_cleaned['Classes'] == 'fire']
# df_not_fire = df_cleaned[df_cleaned['Classes'] == 'not_fire']
# min_class_count = min(len(df_fire), len(df_not_fire))
# df_fire_downsampled = resample(df_fire, replace=False, n_samples=min_class_count, random_state=42)
# df_not_fire_downsampled = resample(df_not_fire, replace=False, n_samples=min_class_count, random_state=42)
# df_balanced = pd.concat([df_fire_downsampled, df_not_fire_downsampled])

### **(B) Oversampling Minority Class (Increase 'fire')**
# df_fire = df_cleaned[df_cleaned['Classes'] == 'fire']
# df_not_fire = df_cleaned[df_cleaned['Classes'] == 'not_fire']
# df_fire_upsampled = resample(df_fire, replace=True, n_samples=len(df_not_fire), random_state=42)
# df_balanced = pd.concat([df_fire_upsampled, df_not_fire])

### **(C) Use Raw Data Without Balancing (No Sampling)**
df_balanced = df_cleaned.copy()

print("Balanced class distribution:")
print(df_balanced['Classes'].value_counts())

x = df_balanced.drop(columns=['Classes', 'day'])
y = df_balanced['Classes']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

k_values = [1, 3, 5, 15, 27, 55, 80, 120, 130, 144, 150]
accuracies, precision_scores, recall_scores, f1_scores = [], [], [], []

for k in k_values:
    model = KNN(k=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    precision_scores.append(report_dict['weighted avg']['precision'])
    recall_scores.append(report_dict['weighted avg']['recall'])
    f1_scores.append(report_dict['weighted avg']['f1-score'])
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', label='accuracy')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.title('KNN Accuracy vs. k Value')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
bar_width = 0.2
indices = np.arange(len(k_values))
plt.bar(indices, accuracies, width=bar_width, label='accuracy', alpha=0.7)
plt.bar(indices + bar_width, precision_scores, width=bar_width, label='precision', alpha=0.7)
plt.bar(indices + 2 * bar_width, recall_scores, width=bar_width, label='recall', alpha=0.7)
plt.bar(indices + 3 * bar_width, f1_scores, width=bar_width, label='f1-score', alpha=0.7)
plt.xlabel('k value')
plt.ylabel('score')
plt.title('Accuracy, Precision, Recall & F1-score vs. k Value')
plt.xticks(indices + bar_width, k_values)
plt.legend()
plt.show()
