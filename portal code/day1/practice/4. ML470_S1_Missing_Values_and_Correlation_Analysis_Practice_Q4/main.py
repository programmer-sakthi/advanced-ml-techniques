import pandas as pd
import warnings
import os
import sys

# Suppress warnings
warnings.simplefilter(action="ignore")

# Input CSV file name
filename = os.path.join(sys.path[0], input())

try:
    # Load CSV file
    df = pd.read_csv(filename)

    # Check for empty or invalid file
    if df.empty:
        print(f"Error: File '{filename}' is empty or invalid.")
        exit()

    # -------- 1. Rows with missing values --------
    print("Rows with missing values (if any):")

    missing_rows = df[df.isnull().any(axis=1)]

    if missing_rows.empty:
        print("No missing values found in the dataset.\n")
    else:
        print(missing_rows)
        print()

    # -------- 2. Correlation matrix --------
    print("Correlation matrix of numeric columns:")

    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()

    print(corr_matrix)

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception:
    print(f"Error: File '{filename}' is empty or invalid.")
