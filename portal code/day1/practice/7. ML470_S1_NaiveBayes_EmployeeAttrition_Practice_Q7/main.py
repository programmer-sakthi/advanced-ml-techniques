import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Read CSV filename from user
filename = os.path.join(sys.path[0], input())

# Check if file exists
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Convert categorical columns to dummy variables
df_encoded = pd.get_dummies(df, columns=["Department", "salary"])

# Separate features and target
X = df_encoded.drop("left", axis=1)
y = df_encoded["left"]

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize and train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Print trained model
print(model)
print()

# Make predictions on test data
predictions = model.predict(X_test)

# Print predictions
print(predictions)
