from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import pickle
import os
import re

def preprocess_reviews(reviews):
    #Process the text to remove unnecessary symbols
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

if __name__ == "__main__":
    filename = "final_model.sav"
    final_model = pickle.load(open(filename, 'rb'))

    stop_words = ['in', 'of', 'at', 'a', 'the']
    cv = CountVectorizer(binary=True, ngram_range=(1, 2), stop_words=stop_words)
    reviews_train = []
    for line in open('movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())
    reviews_train_clean = preprocess_reviews(reviews_train)
    cv.fit(reviews_train_clean)

    predict_text = []
    for line in open('positive.txt', 'r'):
        predict_text.append(line.strip())
    for line in open('negative.txt', 'r'):
        predict_text.append(line.strip())
    predict_text_clean = preprocess_reviews(predict_text)
    X_pred = cv.transform(predict_text_clean)
    y_pred = final_model.predict(X_pred)
    print(y_pred)
