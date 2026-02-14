import pandas as pd
from sklearn.model_selection import train_test_split

data_frame_kidney = pd.read_csv('kidney_disease.csv')

X = data_frame_kidney.drop('classification', axis=1)
y = data_frame_kidney['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# My answers
# We should not train and test a model on the same data because it leads to overfitting[cite: 54].
# The model might just memorize the training examples instead of learning general patterns.
# The purpose of the testing set is to evaluate how the model performs on new, unseen data[cite: 55].