import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

nltk.download("stopwords")

data = pd.read_csv("C:/Users/user/Desktop/INDOLIKE/2/movie.csv")

print("Columns in dataset:", data.columns)

data = data[["review", "sentiment"]].dropna()

print("\nClass Distribution Before Balancing:")
print(data["sentiment"].value_counts())

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["clean_text"] = data["review"].apply(preprocess_text)

vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"), max_features=5000)
X = vectorizer.fit_transform(data["clean_text"])
y = data["sentiment"]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE:")
print(pd.Series(y_train).value_counts())

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))