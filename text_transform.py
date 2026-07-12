import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

tokenize = nltk.word_tokenize
stem = ps.stem

@st.cache_data
def transformed(text):
    if not isinstance(text, str):
        return ""

    words = [
        stem(word)
        for word in tokenize(text.lower())
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(words)
