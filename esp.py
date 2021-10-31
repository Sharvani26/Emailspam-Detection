from os import walk
from string import punctuation
from random import shuffle
from collections import Counter

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

def extract_features(mail_dir1,mail_dir2, vocab):
    files1 = [os.path.join(mail_dir1, fi) for fi in os.listdir(mail_dir1)]
    files2 = [os.path.join(mail_dir2, fi) for fi in os.listdir(mail_dir2)]
    files = files1 + files2
    docId = 0
    matrix = np.zeros((len(files),3000))
    for file in files:
        with codecs.open(file,'r',encoding='utf-8',errors='ignore') as m:
            for i, line in enumerate(m):
                z=m.readlines()
                for i in range(1,len(z)):
                    words = z[i].split()
                    wordId = 0
                    for word in words:
                        if word not in vocab.keys():
                            continue
                        wordId = vocab[word]
                        matrix[docId, wordId] = words.count(word)
            docId += 1
    
    return matrix

words=extract_features(train_dir1,train_dir2,vocab)

train_labels = np.zeros(5172)
train_labels[0:1499]=1

model1 = MultinomialNB() 
model2 = LinearSVC() 
model3 = RandomForestClassifier(n_estimators=100)
model4 = KNeighborsClassifier(n_neighbors=3)
model5 = LogisticRegression()
model1.fit(words,train_labels) 
model2.fit(words,train_labels)
model3.fit(words,train_labels) 
model4.fit(words,train_labels) 
model5.fit(words,train_labels) 

pickle.dump(model1,open("model1.pkl","wb"))
pickle.dump(model2,open("model2.pkl","wb"))
pickle.dump(model3,open("model3.pkl","wb"))
pickle.dump(model4,open("model4.pkl","wb"))
pickle.dump(model5,open("model5.pkl","wb"))