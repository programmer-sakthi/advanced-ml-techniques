import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.simplefilter(action="ignore")

# Input CSV filename
filename = input().strip()

try:
    # Load dataset
    df = pd.read_csv(filename)

    # ---------- 1. Create dummy variables for salary ----------
    salary_dummies = pd.get_dummies(df["salary"], prefix="salary")
    df_salary = pd.concat([df, salary_dummies], axis=1)

    print("Creating dummy variables for salary:")
    print(df_salary.head())
    print()

    # ---------- 2. Create dummy variables for department ----------
    dept_dummies = pd.get_dummies(df_salary["Department"], prefix="dept")
    df_final = pd.concat([df_salary, dept_dummies], axis=1)

    print("Creating dummy variables for department:")
    print(df_final.head())
    print()

    # ---------- 3. Final dataframe ----------
    print("Final dataframe with dummy variables:")
    print(df_final.head())
    print()

    # ---------- 4. Train-test split (70-30) ----------
    train_df, test_df = train_test_split(df_final, train_size=0.7)

    print("Size of training dataset: ", train_df.shape)
    print("Size of test dataset: ", test_df.shape)
    print()

    # ---------- 5. Separate features and target ----------
    X_train = train_df.drop(columns="left", axis=1)
    y_train = train_df["left"]

    X_test = test_df.drop(columns="left", axis=1)
    y_test = test_df["left"]

    print("Shapes of input/output features after train-test split:")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
