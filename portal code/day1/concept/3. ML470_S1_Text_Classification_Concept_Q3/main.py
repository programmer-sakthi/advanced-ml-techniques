import pandas as pd
import os
import sys
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Suppress warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Input CSV file name
filename = os.path.join(sys.path[0], input())

try:
    # Load dataset

    data = pd.read_csv(filename)

    # Extract features and target
    X = data["HealthText"]
    y = data["Outcome"]

    # Convert text to numeric vectors
    vectorizer = CountVectorizer()
    X_vectors = vectorizer.fit_transform(X)

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors, y, test_size=0.2, random_state=42
    )

    # Train Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # New sample health description
    new_sample = [
        "Age group: Senior | BMI status: Overweight | Glucose category: Very High Glucose Level"
    ]

    # Transform new sample using same vectorizer
    new_sample_vector = vectorizer.transform(new_sample)

    # Predict outcome
    prediction = model.predict(new_sample_vector)[0]

    # Output result
    print(f"Prediction: {prediction}")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
