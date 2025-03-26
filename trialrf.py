import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from randomForest import RandomForest

file_path_cleaned = r"C:\Users\PureTech\Desktop\greymatter\youth destroyer\Literature\Article material\Algerian_forest_fires_dataset_CLEANED.csv"
df_cleaned = pd.read_csv(file_path_cleaned)
df_cleaned['Classes'] = df_cleaned['Classes'].str.strip().str.lower()
df_cleaned['Classes'] = df_cleaned['Classes'].replace({'fire': 'fire', 'not fire': 'not_fire'})
df_cleaned = df_cleaned[df_cleaned['Classes'].isin(['fire', 'not_fire'])]
print("Original class distribution:")
print(df_cleaned['Classes'].value_counts())

# **(A) Undersampling Majority Class (Reduce 'not_fire')**
df_fire = df_cleaned[df_cleaned['Classes'] == 'fire']
df_not_fire = df_cleaned[df_cleaned['Classes'] == 'not_fire']
min_class_count = min(len(df_fire), len(df_not_fire))
df_fire_downsampled = resample(df_fire, replace=False, n_samples=min_class_count)
df_not_fire_downsampled = resample(df_not_fire, replace=False, n_samples=min_class_count)
df_balanced = pd.concat([df_fire_downsampled, df_not_fire_downsampled])

# ## **(B) Oversampling Minority Class (Increase 'fire')**
# df_fire = df_cleaned[df_cleaned['Classes'] == 'fire']
# df_not_fire = df_cleaned[df_cleaned['Classes'] == 'not_fire']
# df_fire_upsampled = resample(df_fire, replace=True, n_samples=len(df_not_fire))
# df_balanced = pd.concat([df_fire_upsampled, df_not_fire])

### **(C) Use Raw Data Without Balancing (No Sampling)**
# df_balanced = df_cleaned.copy()

print("Balanced class distribution:")
print(df_balanced['Classes'].value_counts())

X = df_balanced.drop(columns=['Classes', 'day'])
y = df_balanced['Classes'].map({'fire': 1, 'not_fire': 0})
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
y_train = np.array(y_train)
y_test = np.array(y_test)

n_trees_values = [1, 3, 5, 15, 27, 55, 80, 120, 130, 144, 150]
accuracies, precision_scores, recall_scores, f1_scores = [], [], [], []

for n_trees in n_trees_values:
    model = RandomForest(n_estimators=n_trees)
    model.fit(x_train.to_numpy(), y_train)

    y_pred = model.predict(x_test.to_numpy())
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    precision_scores.append(report_dict['weighted avg']['precision'])
    recall_scores.append(report_dict['weighted avg']['recall'])
    f1_scores.append(report_dict['weighted avg']['f1-score'])


plt.figure(figsize=(8, 5))
plt.plot(n_trees_values, accuracies, marker='o', linestyle='-', label='accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy vs. Number of Trees')
plt.xticks(n_trees_values)
plt.grid(True)
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
bar_width = 0.2
indices = np.arange(len(n_trees_values))

plt.bar(indices, accuracies, width=bar_width, label='Accuracy', alpha=0.7)
plt.bar(indices + bar_width, precision_scores, width=bar_width, label='Precision', alpha=0.7)
plt.bar(indices + 2 * bar_width, recall_scores, width=bar_width, label='Recall', alpha=0.7)
plt.bar(indices + 3 * bar_width, f1_scores, width=bar_width, label='F1-score', alpha=0.7)

plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.title('Accuracy, Precision, Recall & F1-score vs. Number of Trees')
plt.xticks(indices + bar_width, n_trees_values)
plt.legend()
plt.show()
