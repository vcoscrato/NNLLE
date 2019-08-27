import numpy as np
import pandas as pd
from pprint import pprint
from time import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit

from matplotlib import pyplot as plt

from nnlocallinear import NLS, LLS


class AmazonData(object):

    def __init__(self, file, target, features):
        self.df = pd.read_csv(file)
        self.target = target
        self.features = features
        # Train data to be defined.
        self.x_train = None
        self.y_train = None
        # Test data to be defined.
        self.x_test = None
        self.y_test = None
        # Statistic from kfold
        self.statistic_kfold = None
        self.train_index = None
        self.test_index = None
        # Balancing the data.
        self.information = None
        self.df_balanced = None

    def prepare_data(self):
        self.statistic_kfold = pd.DataFrame()
        self.df = self.df.sample(frac=1, random_state=0)
        self.df['bins_target'] = np.ceil(self.df[self.target])
        self.statistic_kfold['whole_data'] = self.df['bins_target'].value_counts() / self.df.shape[0]

    def show_histogram_target(self):
        self.df['bins_target'].hist()
        plt.show()

    @staticmethod
    def stratified_split(df, random_state=0, test_size=0.2):
        stratied_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        for train_index, test_index in stratied_split.split(df, df['bins_target']):
            return train_index, test_index

    def under_sampling(self, max_instances=None):
        if self.information is None:
            unique_values = np.unique(self.df[self.target].values)
            information = {}
            for value in unique_values:
                selection = (self.df[self.target] == value).values
                count = np.sum(selection)
                information[value] = {'selection': selection, 'count': count}
            self.information = information

        lower_boundary = np.min([e['count'] for e in self.information.values()])
        if max_instances is not None:
            if lower_boundary < max_instances:
                print('Warning: number of instances available for under-sampling is lower than max_intances!')
            else:
                lower_boundary = max_instances

        self.df_balanced = pd.DataFrame()
        for piece in self.information.values():
            selection = piece['selection']
            self.df_balanced = self.df_balanced.append(self.df[selection].sample(n=lower_boundary))

        self.df_balanced = self.df_balanced.sample(frac=1.0)
        indices_train, indices_test = self.stratified_split(self.df_balanced)
        df_train = self.df_balanced.iloc[indices_train]
        df_test = self.df_balanced.iloc[indices_test]
        self.x_train = df_train[self.features]
        self.y_train = df_train[self.target]

        self.x_test = df_test[self.features]
        self.y_test = df_test[self.target]


if __name__ == "__main__":
    data = AmazonData(
        file='../data/amazon-fine-food-reviews/Reviews.csv'
        , target='Score'
        , features='Text'
    )
    data.prepare_data()
    data.under_sampling(max_instances=10)

    pipeline = Pipeline([
        ('vectoring', CountVectorizer()),
        ('tf_idf_normalize', TfidfTransformer()),
        ('model', NLS(
            verbose=0
            , es=True
            , gpu=False
            , scale_data=False
            , varying_theta0=False
            , fixed_theta0=True
            , dataloader_workers=0
            # , with_mean=False
        )),
    ])

    parameters = {
        'vectoring__max_df': (0.5, 0.75, 1.0)
        # Options to use uni-grams or bi-grams
        , 'vectoring__ngram_range': ((1, 1), (1, 2))
        # 'vectoring__max_features': (None, 5000, 10000, 50000),
        # 'tf_idf_normalize__use_idf': (True, False),
        # 'tf_idf_normalize__norm': ('l1', 'l2'),

        # Parameters for NLS
        , 'model__es_give_up_after_nepochs': [500]
        , 'model__hidden_size': [100, 250, 500]
        , 'model__num_layers': [1, 3, 5]
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()

    vectoring = CountVectorizer(max_df=0.5, ngram_range=(1, 1))
    x_train = vectoring.fit_transform(data.x_train)
    tf_idf_normalize = TfidfTransformer()
    x_train = tf_idf_normalize.fit_transform(x_train)

    # print(x_train.toarray())
    # print(x_train.toarray().shape)
    # np.savetxt(fname='test.txt', X=x_train.toarray())

    model = NLS(
            verbose=0
            , es=True
            , gpu=False
            , scale_data=False
            , varying_theta0=False
            , fixed_theta0=True
            , dataloader_workers=0
            , es_give_up_after_nepochs=500
            , hidden_size=100
            , num_layers=1
        )
    print(data.y_train.values.reshape(-1, 1))
    model.fit(x_train, data.y_train.values.reshape(-1, 1))
    exit()
    grid_search.fit(data.x_train.values, data.y_train.values)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
