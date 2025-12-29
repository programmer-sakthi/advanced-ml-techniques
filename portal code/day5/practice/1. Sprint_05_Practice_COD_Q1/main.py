import pandas as pd
import numpy as np
import os
import sys
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# 1. File Input and Loading

filename = os.path.join(sys.path[0], input())


try:
    df = pd.read_csv(filename)
except Exception:
    print(f"Error: Unable to read file '{filename}'.")
    exit()


# 2. Categorical Variable Encoding

label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# 3. Outlier Analysis (IQR)

print("--- Outlier Assessment ---")

for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"{col}: {outliers} outliers")

        # -----------------------------
        # 4. Outlier Treatment (Capping)
        # -----------------------------
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])


# 5. Feature-Target Separation

target_col = "Purchase Likelihood"

if target_col not in df.columns:
    print("Error: Target column 'Purchase Likelihood' not found in dataset.")
    exit()

X = df.drop(columns=[target_col])
y = df[target_col]


# 6. Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# -----------------------------
# 7. Train-Test Split (80-20, Sequential)
# -----------------------------
split_index = int(0.8 * len(df))

X_train = X_scaled.iloc[:split_index]
X_test = X_scaled.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# -----------------------------
# 8. Model Training and Evaluation
# -----------------------------
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# -----------------------------
# 9. Final Output
# -----------------------------
accuracy = round(accuracy, 2)
print("\n==============================")
print(f"Model Accuracy: {accuracy} %")
print("==============================")
