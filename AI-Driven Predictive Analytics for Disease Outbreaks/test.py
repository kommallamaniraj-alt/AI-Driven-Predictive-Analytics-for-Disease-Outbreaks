import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
import os
from keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import ADASYN

from nltk.corpus import stopwords
import nltk
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

dataset = pd.read_csv("data.csv", encoding='iso-8859-1')
outbreak = dataset['Outbreak']
labels, counter = np.unique(outbreak, return_counts = True)
dataset['Temperature'] = dataset['Temperature'].fillna(value=dataset['Temperature'].mean())
dataset['Humidity'] = dataset['Humidity'].fillna(value=dataset['Humidity'].mean())

temperature = dataset['Temperature'].ravel()
humidity = dataset['Humidity'].ravel()

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

def getLabel(name):
    index = -1
    count = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            count = counter[i]
            break
    return index, count

desc = dataset['Description']
X = []
Y = []
temp = []
hum= []
'''
for i in range(len(desc)):
    news = str(desc[i])
    news = news.strip('\n').strip().lower()
    if len(news) > 0:
        news = cleanText(news)
        label, count = getLabel(outbreak[i])
        if count < 6:
            for k in range(0, 15):
                X.append(news)
                Y.append(label)
                temp.append(temperature[i])
                hum.append(humidity[i])
        else:
            X.append(news)
            Y.append(label)
            temp.append(temperature[i])
            hum.append(humidity[i])
        print(str(len(X))+" "+str(label))

temperature = np.asarray(temp)
humidity = np.asarray(hum)
X = np.asarray(X)
Y = np.asarray(Y)
np.save("model/environment", np.asarray([temperature, humidity]))
np.save("model/X", X)
np.save("model/Y", Y)
'''
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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
'''
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)
data = np.asarray([X_train, X_test, y_train, y_test])
np.save("model/data", data)
'''
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

X_train1 = np.reshape(X_train, (X_train.shape[0], 28, 28, 3))
X_test1 = np.reshape(X_test, (X_test.shape[0], 28, 28, 3))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train1, y_train1, batch_size = 32, epochs = 100, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")

predict = cnn_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)
    


param_grid = {'n_estimators': [100], 'max_depth': [2, 3]}

tune_rf = GridSearchCV(RandomForestClassifier(), param_grid)
tune_rf.fit(X_train, y_train)
predict = tune_rf.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)



param_grid = {'min_samples_split': [0.2, 0.5, 0.7], 'criterion': ["gini"]}
tune_dt = GridSearchCV(DecisionTreeClassifier(), param_grid)
tune_dt.fit(X_train, y_train)
predict = tune_dt.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

    











