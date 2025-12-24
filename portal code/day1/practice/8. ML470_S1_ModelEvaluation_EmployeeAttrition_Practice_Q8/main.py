import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import ML_Modules

# Read CSV filename
filename = os.path.join(sys.path[0], input())

# File existence check
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# One-hot encoding categorical columns
df_encoded = pd.get_dummies(df, columns=["Department", "salary"])

# Split features and target
X = df_encoded.drop("left", axis=1)
y = df_encoded["left"]

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
ML_Modules.evaluate_classifier(y_test, y_pred)

# Confirmation message for AUC-ROC function
print("code is available inside the 'ML_Modules.py' file")
