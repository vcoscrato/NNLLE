import sklearn
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt

from nnlocallinear import NLS, LLS, NNPredict

import lime
from lime import lime_text
from sklearn.pipeline import make_pipeline

import numpy as __numpy


categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
train_vectors = train_vectors.toarray()
test_vectors = test_vectors.toarray()

names_features = np.array(vectorizer.get_feature_names())


def train_NN():
    parameters = {
            'model__es_give_up_after_nepochs': [20]
            , 'model__hidden_size': [100, 250, 500]
            , 'model__num_layers': [1, 3, 5]
        }

    comb_parameters = [{
            'es_give_up_after_nepochs': 5  # TODO: 20
            , 'hidden_size': 5 # 20
            , 'num_layers': 1
            , 'n_classification_labels': 2,

        }
            ]

    for parameter in comb_parameters:
        model = NNPredict(
            verbose=1
            , es=True
            , gpu=True
            , scale_data=False
            , varying_theta0=False
            , fixed_theta0=True
            , dataloader_workers=0
            # , with_mean=False
            , **parameter
        )
        model.fit(x_train=train_vectors, y_train=newsgroups_train.target)
    return model

def train_RF():
    import sklearn.ensemble
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    model.fit(train_vectors, newsgroups_train.target)
    return model




from sklearn.utils import check_array
from scipy.sparse import issparse

class Validate(object):

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, x):
         # return self._validate_X_predict(x)
        return x.toarray()

    def _validate_X_predict(self, X, check_input=True):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        # if self.n_features_ != n_features:
        #     raise ValueError("Number of features of the model must "
        #                      "match the input. Model n_features is %s and "
        #                      "input n_features is %s "
        #                      % (self.n_features_, n_features))

        return X


# model = train_RF()
model = train_NN()
pred = model.predict(test_vectors)
print('Score:', sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary'))

val = Validate()
c = make_pipeline(vectorizer, val, model)

idx = 83
# x = vectorizer.transform([newsgroups_test.data[idx]]).toarray()
# model.predict(x)

c.predict_proba([newsgroups_test.data[idx]])