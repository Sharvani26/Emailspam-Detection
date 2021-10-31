from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np

import pickle
import pandas as pd
import sklearn as sk
import nltk
import numpy as np
from sklearn.datasets import load_files
from collections import Counter
import codecs

import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

model1=pickle.load(open("model1.pkl","rb"))
model2=pickle.load(open("model2.pkl","rb"))
model3=pickle.load(open("model3.pkl","rb"))
model4=pickle.load(open("model4.pkl","rb"))
model5=pickle.load(open("model5.pkl","rb"))

@app.route("/")
def spam():
    return render_template("Emailspam.html")

@app.route("/predict",methods=["post"])
def ham():

    m=request.form["spam"]

    train_dir1 = "C:/Users/sharv/OneDrive/Desktop/enron1/spam"
    train_dir2 = "C:/Users/sharv/OneDrive/Desktop/enron1/ham"

    def make_vocabulary(train_dir):
        emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
        all_words = []
        for mail in emails:
            with codecs.open(mail,'r',encoding='utf-8',errors='ignore') as m:
                z=m.readlines()
                for i in range(1,len(z)):
                    words = z[i].split()
                    all_words += words
    
        dictionary = Counter(all_words)

        list_to_remove = list(dictionary.keys())

        for item in list_to_remove:
            if item.isalpha() == False:
                del dictionary[item]
            elif len(item) == 1:
                del dictionary[item]
        return dictionary

    dictionary1=make_vocabulary(train_dir1)
    dictionary2=make_vocabulary(train_dir2)
    dictionary=dictionary2+dictionary1
    dictionary = dictionary.most_common(3000)

    vocab = {}
    for i, (word, frequency) in enumerate(dictionary):
        vocab[word] = i
    
    docId = 0
    matrix = np.zeros((1,3000))
    words = m.split()
    wordId = 0
    for word in words:
        if word not in vocab.keys():
            continue
        wordId = vocab[word]
        matrix[docId, wordId] = words.count(word)

    words1=matrix

    z=[]
    result1 = model1.predict(words1)
    result2 = model2.predict(words1)
    result3 = model3.predict(words1)
    result4 = model4.predict(words1)
    result5 = model5.predict(words1)
    z.append(result1[0])
    z.append(result2[0])
    z.append(result3[0])
    z.append(result4[0])
    z.append(result5[0])

    true=0
    false=0
    for i in z:
        if i==1.0:
            true=true+1
        else:
            false=false+1
    if(true>false):
        return render_template("Emailspam.html",h="Input Email is Spam")
    else:
        return render_template("Emailspam.html",h="Input Email is not Spam")

if __name__== "__main__":
    app.run(debug=True)