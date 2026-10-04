
import streamlit as st
import joblib
import re
import nltk

# Download required NLTK data
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Load Model and Vectorizer
# -----------------------------

model = joblib.load("news_category_svm.pkl")
tfidf = joblib.load("news_category_tfidf.pkl")

stop_words = set(stopwords.words("english"))

# -----------------------------
# Text Preprocessing
# -----------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    words = [
        word for word in text.split()
        if word not in stop_words and len(word) > 2
    ]

    return ' '.join(words)

# -----------------------------
# Application Interface
# -----------------------------

st.title("📰 News Article Category Classifier")

st.write(
    "Enter a news article or headline below and the machine learning "
    "model will predict its category."
)

st.info(
    "Model: Linear SVM | Features: TF-IDF Unigrams + Bigrams"
)

article = st.text_area(
    "Enter News Article",
    placeholder="Example: The government announced new economic policies...",
    height=180
)

if st.button("Predict Category"):

    if article.strip() == "":
        st.warning("Please enter a news article first.")

    else:
        cleaned_text = preprocess_text(article)

        article_tfidf = tfidf.transform([cleaned_text])

        prediction = model.predict(article_tfidf)[0]

        st.success(f"Predicted Category: **{prediction}**")

        with st.expander("View Preprocessed Text"):
            st.write(cleaned_text)

# -----------------------------
# Project Information
# -----------------------------

st.divider()

st.caption(
    "News Article Categorization and Classification | "
    "TF-IDF + Linear SVM"
)
