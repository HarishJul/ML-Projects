from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    #Loading and pre-processing
    sms = pd.read_csv('spam.csv', encoding = 'latin-1')
    sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1)
    sms.columns = ['label','message']
    sms['label'] = sms['label'].map({'ham': 0, 'spam': 1})
    
    #text pre-processing function
    def pre_process_message(message, lower_case = True, stem = True, stop_words = True):
        if lower_case:
            message = message.lower()
        words = word_tokenize(message)
        words = [w for w in words if len(w) > 2]
        if stop_words:
            sw = stopwords.words('english')
            words = [w for w in words if w not in sw]
        if stem:
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
        return " ".join(words)
    
    #Model building
    X = sms['message'].apply(pre_process_message)
    y = sms['label']
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train,y_train)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug = True)