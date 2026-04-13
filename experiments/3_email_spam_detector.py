# Predict whether an email is spam or not based on its content.
# This is a classification problem.
# Models cannot understand NLP (text) directly, we need to convert it into numbers.

# Steps:
# 1. Convert the text into numbers (feature extraction).
# 2. Train a machine learning model on the features.    
# 3. Evaluate the model. Predict spam ( 1 or 0 ).
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import joblib

def load_data(path):
    return pd.read_csv(path)

def train_model(df):
    X = df["text"]
    y = df["spam"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(
        CountVectorizer(), # Convert text to vectors (numbers)
        MultinomialNB() # Train a Naive Bayes classifier on the vectors
    )

    # Train Model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    # Save the model
    joblib.dump(model, "email_spam_detector_model.pkl")

    return model, accuracy, cm

def predict_email(model, email):
    return model.predict([email])[0]

df = load_data("../data/emails.csv")
model, accuracy, cm = train_model(df)

print("Accuracy:", accuracy)
emailPrediction = predict_email(model, "Congratulations! You've won a free ticket to the Bahamas. Click here to claim.")
print("Prediction:", emailPrediction)
print("Confusion Matrix:\n", cm)