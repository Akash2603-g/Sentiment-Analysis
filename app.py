import streamlit as st
import joblib

# Load trained sentiment model and vectorizer
model = joblib.load("sentiment_model.pkl")  
vectorizer = joblib.load("vectorizer.pkl")  

st.title("Sentiment Analysis App")
st.write("Type a sentence and find out if it's Positive, Negative, or Neutral!")

# User input
user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Transform text
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]

        # Display result
        if prediction == "positive":
            st.success("😊 Sentiment: Positive")
        elif prediction == "negative":
            st.error("😠 Sentiment: Negative")
        else:
            st.info("😐 Sentiment: Neutral")
    else:
        st.warning("Please enter some text!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset (make sure you have Tweets.csv in the same folder)
df = pd.read_csv("Tweets.csv")

# For simplicity, use only 'text' and 'airline_sentiment' columns
X = df['text']
y = df['airline_sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved!")
