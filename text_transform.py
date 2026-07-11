import string
from nltk.corpus import stopwords
import string,nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = nltk.stem.porter.PorterStemmer()

stop_words = set(stopwords.words('english'))

def transformed(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)


