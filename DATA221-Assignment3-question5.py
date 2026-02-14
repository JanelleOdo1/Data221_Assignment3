import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_frame_kidney = pd.read_csv('kidney_disease.csv')
X = data_frame_kidney.drop('classification', axis=1)
y = data_frame_kidney['classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

k_values = [1, 3, 5, 7, 9]
print("k Value | Test Accuracy")
print("-----------------------")

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(k, accuracy)

# My answers
# Changing the value of k affects how complex or simple the model's decision-making is.
# Very small values of k (like k=1) may cause overfitting because the model is too sensitive to noise.
# Very large values of k may cause underfitting because the model ignores local patterns and becomes too simple.