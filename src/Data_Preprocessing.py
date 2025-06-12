import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def tokenization(text):
    text = text.lower()  # Convert text to lowercase for uniformity
    text = text.strip()  # Remove leading/trailing whitespace
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation marks
    # Remove stop words (common words like 'the', 'is', etc.)
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove non-alphabetic characters (numbers, symbols)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    # Apply stemming to reduce words to their root form (e.g., 'running' -> 'run')
    text = ' '.join([stemmer.stem(word) for word in text.split()])  
    text = word_tokenize(text)  # Tokenize text into list of words
    return text
