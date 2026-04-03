import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required resources (run once)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

text = "Natural Language Processing is fun. It is widely used in artificial intelligence."

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentence_tokenization", sentences)

# Word Tokenization
tokens = word_tokenize(text.lower())
print("word tokenization", tokens)

# Remove stopwords and punctuation
stop_words = set(stopwords.words('english'))
print(f"Stop words: {stop_words}")
tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
print("remove stopwords and punctuation", tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(t) for t in tokens]
print(stemmed_tokens)
