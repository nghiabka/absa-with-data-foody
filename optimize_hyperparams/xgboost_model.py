import pickle
import warnings
from os.path import join, dirname

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.load_data import load_dataset
from src.model import XGboostModel

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    X_train = pickle.load(open("../data/X_train.pkl", "rb"))
    y_train = pickle.load(open("../data/y_train.pkl", "rb"))

    X_dev = pickle.load(open("../data/X_dev.pkl", "rb"))
    y_dev = pickle.load(open("../data/y_dev.pkl", "rb"))

    X_test = pickle.load(open("../data/X_test.pkl", "rb"))
    y_test = pickle.load(open("../data/y_test.pkl", "rb"))

    models = []
    for n_iter in [ 200, 500]:
        # for max_depth in [50, 100, 140, 160, 200, 300]:
        for max_depth in [ 400, 500]:
            # for max_features in [1000, 2000, 2200, 2400, 2600, 3000]:
            for max_features in [3000, 4000]:
                name = "(n_iter {0} max_depth {1}) + Count(bigram, max_features {2})".format(n_iter, max_depth,
                                                                                                    max_features)
                params = {"n_iter": n_iter, "max_depth": max_depth}
                model = XGboostModel(
                    name,
                    params,
                    CountVectorizer(ngram_range=(1, 2), max_features=max_features)
                )
                models.append(model)
    # for n_iter in [100, 140, 160, 200, 500]:
    for n_iter in [200, 500]:

        # for max_depth in [50, 100, 140, 160, 200, 300]:
        for max_depth in [ 400, 500]:
            # for max_features in [1000, 2000, 2200, 2400, 2600, 3000]:
            for max_features in [ 3000, 4000]:
                name = "(n_iter {0} max_depth {1}) + Tfidf(bigram, max_features {2})".format(n_iter, max_depth,
                                                                                                    max_features)
                params = {"n_iter": n_iter, "max_depth": max_depth}
                model = XGboostModel(
                    name,
                    params,
                    TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
                )
                models.append(model)

    for model in models:
        model.load_data(X_train, y_train, X_dev, y_dev)
        model.fit_transform()
        model.train()
        model.evaluate(model_name="XGBoost")
