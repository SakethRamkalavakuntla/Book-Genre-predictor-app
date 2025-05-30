import pickle
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask, request, render_template

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Clean the text
def cleantext(text):
    text = re.sub(r"'\''", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    return text.lower()

# Remove stopwords
def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

# Lemmatize
def lemmatizing(text):
    lemma = WordNetLemmatizer()
    return ' '.join(lemma.lemmatize(word) for word in text.split())

# Stem
def stemming(text):
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())

# Full preprocessing + prediction
def test(text, model, tfidf_vectorizer):
    text = cleantext(text)
    text = removestopwords(text)
    text = lemmatizing(text)
    text = stemming(text)
    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)

    newmapper = {
        0: 'Fantasy',
        1: 'Science Fiction',
        2: 'Crime Fiction',
        3: 'Historical novel',
        4: 'Horror',
        5: 'Thriller'
    }

    return newmapper.get(predicted[0], "Unknown Genre")

# Load model and TF-IDF vectorizer
with open('bookgenremodel1.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidfvector1.pkl', 'rb') as file1:
    tfidf_vectorizer = pickle.load(file1)

# Flask setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        mydict = request.form
        text = mydict.get("summary", "").strip()

        if not text:
            return render_template('index1.html', error="Please enter a valid summary.", showresult=False)

        try:
            prediction = test(text, model, tfidf_vectorizer)
            return render_template('index1.html', genre=prediction, text=text[:100], showresult=True)
        except Exception as e:
            return render_template('index1.html', error=f"Error occurred: {str(e)}", showresult=False)

    return render_template('index1.html', showresult=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
