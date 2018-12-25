import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from AdaSample import AdaSample

HideFrac = 0.8
TrainFrac = 0.6
N = 500000

SampFrac = 0.01 #Subsampling of training data when preforming Adasampling
NBoosts = 50 #Number of AdaSampling rounds

X, y = datasets.make_classification(N, 10, 5)

y_PU = np.copy(y)
Ps = np.where(y_PU == 1)[0]
y_PU[np.random.choice(Ps, int(np.floor(len(Ps) * 0.8)))] = 0

inds = np.random.permutation(X.shape[0])
train_inds, test_inds = inds[:int(N * 0.6)], inds[int(N * 0.6):]

X_train, X_test = X[train_inds, :], X[test_inds, :]
y_train, y_test = y[train_inds], y[test_inds]
y_PU_train, y_PU_test = y_PU[train_inds], y_PU[test_inds]

clf_res = {}
for name, clf in [("SGD_lasso", SGDClassifier(loss="log", penalty="l1")),
                  ("RF", RandomForestClassifier()),
                  ("SVM_SGD", SGDClassifier(loss="modified_huber", penalty="l1")),
                  ("NB", GaussianNB())]:
    print(name)
    clf_res[name] = {}
    ada = AdaSample(clone(clf))
    ada.fit(X_train, y_PU_train, C=10, samp_frac=SampFrac, n_boost=NBoosts)
    probas = ada.predict_proba(X_test, single=True)[:, 1]
    clf_res[name]["PU_single"] = accuracy_score(y_test, (probas > 0.5).astype(np.int))

    probas = ada.predict_proba(X_test)[:, 1]
    clf_res[name]["PU_ensemble"] = accuracy_score(y_test, (probas > 0.5).astype(np.int))

    naive_clf = clone(clf)
    naive_clf.fit(X_train, y_train)
    clf_res[name]["Naive_clf"] = naive_clf.score(X_test, y_test)

    uNeg_clf = clone(clf)
    uNeg_clf.fit(X_train, y_PU_train)
    clf_res[name]["uNeg_clf"] = uNeg_clf.score(X_test, y_test)

print(pd.DataFrame.from_dict(clf_res))
