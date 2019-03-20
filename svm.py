from sklearn.datasets import load_svmlight_files
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np

X_train, y_train, X_test, y_test = load_svmlight_files(("train.feat", "test.feat"))

def train():
    final_model = LogisticRegression(C=0.05)

    l,_ = X_train.shape
    tr_idx = np.arange(l)
    np.random.shuffle(tr_idx)
    X_shuffle = X_train[tr_idx,:]
    target_shuffle = np.array(y_train)[tr_idx]
    final_model.fit(X_shuffle, target_shuffle)

    return final_model

if __name__ == "__main__":
    final_model = train()
    print ("Final Accuracy: %s"
           % accuracy_score(y_test, final_model.predict(X_test)))
