import pickle
from os.path import dirname, join
from time import time

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.load_data import load_dataset


def save_model(filename, clf):
    pickle.dump(clf, open(filename, 'wb'))


serialization_dir = join(dirname(__file__), "snapshots")



print("Load data...")
X_train = pickle.load(open("./data/X_train.pkl","rb"))
y_train = pickle.load(open("./data/y_train.pkl","rb"))


X_dev = pickle.load(open("./data/X_dev.pkl","rb"))
y_dev = pickle.load(open("./data/y_dev.pkl","rb"))


X_test = pickle.load(open("./data/X_test.pkl","rb"))
y_test = pickle.load(open("./data/y_test.pkl","rb"))

X_train = X_train + X_dev
y_train = y_train + y_dev
print(len(y_train))
target_names= list(set(sum([s for s in y_train], [])))
# for target_name in sorted(target_names):
#     print(target_name)
print(target_names)
# print("%d documents" % len(X_train))
print("%d categories" % len(target_names))

print("\nTraining model...")

t0 = time()
transformer = CountVectorizer(ngram_range=(1, 2))
X_train = transformer.fit_transform(X_train)

y_transformer = MultiLabelBinarizer()
y_train = y_transformer.fit_transform(y_train)

model = OneVsRestClassifier(LinearSVC())
estimator = model.fit(X_train, y_train)
t1 = time() - t0
print("Train time: %0.3fs" % t1)

print("\nEvaluate...")
y_dev = y_transformer.transform(y_dev)
X_dev = transformer.transform(X_dev)
y_pred = estimator.predict(X_dev)
print('F1 Score:', np.round(metrics.f1_score(y_dev, y_pred, average='micro'), 3))

print("report: ")
# print(metrics.classification_report(y_true=y_dev,y_pred=y_pred))
print("+====================================================================================")
print("\nSave model...")
t0 = time()
save_model(serialization_dir + "/x_transformer.pkl", transformer)
save_model(serialization_dir + "/y_transformer.pkl", y_transformer)
save_model(serialization_dir + "/model.pkl", estimator)
t1 = time() - t0
print("Save model time: %0.3fs" % t1)