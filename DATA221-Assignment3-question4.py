import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

data_frame_kidney = pd.read_csv('kidney_disease.csv')
X = data_frame_kidney.drop('classification', axis=1)
y = data_frame_kidney['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_prediction = knn.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_prediction))
print("Accuracy:", accuracy_score(y_test, y_prediction))
print("Precision:", precision_score(y_test, y_prediction))
print("Recall:", recall_score(y_test, y_prediction))
print("F1-score:", f1_score(y_test, y_prediction))

# My answers
# In this context, a true positive means correctly identifying a patient with kidney disease.
# A true negative is correctly identifying a healthy person, while false positive is a false alarm.
# A false negative is the most dangerous because it means we missed a real case of disease.
# Accuracy alone may not be enough because if the classes are imbalanced, a model can be wrong often but still look "accurate".
# Recall is the most important criteria here because missing a kidney disease case is very serious for patient health.