from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from keras.models import load_model
import numpy as np
from nltk.corpus import stopwords
import nltk
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd

app = Flask(__name__)

app.secret_key = 'welcome'
global user

#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation) #remove punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]#take only alphabets
    tokens = [w for w in tokens if not w in stop_words]#remove stop words
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens] #apply stemming and lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

dataset = pd.read_csv("Dataset/Outbreak.csv", encoding='iso-8859-1')
outbreak = dataset['Outbreak']
labels, counter = np.unique(outbreak, return_counts = True)
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
environment = np.load("model/environment.npy")
temperature, humidity = environment

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=2350)
X = tfidf_vectorizer.fit_transform(X).toarray()

X = np.column_stack((X, temperature))
X = np.column_stack((X, humidity))
print(X.shape)

scaler = MinMaxScaler((0, 1))
X = scaler.fit_transform(X)

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        global labels, scaler, tfidf_vectorizer
        report = request.form['t1']
        temperature = request.form['t2']
        humidity = request.form['t3']
        temp = []
        temp.append(float(temperature))
        temp = np.asarray(temp)
        hum = []
        hum.append(humidity)
        hum = np.asarray(hum)
        report = report.strip('\n').strip().lower()
        report = cleanText(report)
        report = tfidf_vectorizer.transform([report]).toarray()
        report = np.column_stack((report, temp))
        report = np.column_stack((report, hum))
        report = scaler.transform(report)
        print(report.shape)
        report = np.reshape(report, (report.shape[0], 28, 28, 3))
        cnn_model = load_model("model/cnn_weights.hdf5")
        predict = cnn_model.predict(report)
        predict = np.argmax(predict)
        disease = labels[predict]
        return render_template('Predict.html', data='<font size="3" color="blue">Predicted Disease Outbreak = '+disease+"</font>")

@app.route('/Predict', methods=['GET', 'POST'])
def Predict():
    return render_template('Predict.html', data='')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', data='')

@app.route('/Logout')
def Logout():
    return render_template('index.html', data='')

if __name__ == '__main__':
    app.run()










