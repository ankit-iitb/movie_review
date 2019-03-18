from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os
import re

reviews_train = []
reviews_test = []
def read_reviews():
    for line in open('movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())
    for line in open('movie_data/full_test.txt', 'r'):
        reviews_test.append(line.strip())

def preprocess_reviews(reviews):
    #Process the text to remove unnecessary symbols
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

def train(reviews_train_clean):
    stop_words = ['in', 'of', 'at', 'a', 'the']
    cv = CountVectorizer(binary=True, ngram_range=(1, 2), stop_words=stop_words)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)

    # First 12500 are +ve reviews and last 12500 are -ve reviews
    target = [1 if i < 12500 else 0 for i in range(25000)]
    final_model = LogisticRegression(C=0.05)

    l,_ = X.shape
    tr_idx = np.arange(l)
    np.random.shuffle(tr_idx)
    X_shuffle = X[tr_idx,:]
    target_shuffle = np.array(target)[tr_idx]
    final_model.fit(X_shuffle, target_shuffle)
    return final_model

if __name__ == "__main__":
    read_reviews()
    reviews_train_clean = preprocess_reviews(reviews_train)
    reviews_test_clean = preprocess_reviews(reviews_test)
    final_model = train(reviews_train_clean)

    # Check accurancy
    stop_words = ['in', 'of', 'at', 'a', 'the']
    cv = CountVectorizer(binary=True, ngram_range=(1, 2), stop_words=stop_words)
    cv.fit(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)
    target = [1 if i < 12500 else 0 for i in range(25000)]
    print ("Final Accuracy: %s"
           % accuracy_score(target, final_model.predict(X_test)))


    predict_text = []
    for line in open('positive.txt', 'r'):
        predict_text.append(line.strip())
    for line in open('negative.txt', 'r'):
        predict_text.append(line.strip())
    predict_text_clean = preprocess_reviews(predict_text)
    X_pred = cv.transform(predict_text_clean)
    y_pred = final_model.predict(X_pred)
    print(y_pred)
