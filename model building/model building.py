import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# Text preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r"''", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = ' '.join(text.split())
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def lemmatize_text(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return ' '.join(lemmatized)

def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Load dataset
books = pd.read_csv('BooksDataSet.csv')
books = books[['book_id', 'book_name', 'genre', 'summary']]

# Apply text preprocessing
books['summary'] = books['summary'].apply(clean_text)
books['summary'] = books['summary'].apply(lemmatize_text)
books['summary'] = books['summary'].apply(stem_text)

# Encode genres
genre_list = books['genre'].unique().tolist()
mapper = {genre: idx for idx, genre in enumerate(genre_list)}
books['genre_encoded'] = books['genre'].map(mapper)

# Split dataset
X = books['summary']
y = books['genre_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=557)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train MultinomialNB model
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
model_path = 'bookgenremodel1.pkl'
vectorizer_path = 'tfidfvector1.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model_nb, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

model_path, vectorizer_path
