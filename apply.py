import numpy as np
from nnlocallinear import NNPredict, LLE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from time import time
import pickle
import os

amazon = pd.read_csv('/home/vcoscrato/Datasets/amazon fine foods.csv')[0:100000]
bow = CountVectorizer(stop_words='english', min_df=0.02, max_df=0.5)
x = bow.fit_transform(amazon['Text']).toarray()
features = np.array(bow.get_feature_names())
x = x[:, features != 'br']
features = features[features != 'br']
x_train, x_test, y_train, y_test = train_test_split(x, amazon['Score'], test_size=0.2, random_state=0)

def apply(n_grid, NN_arch, LLE_var):
    MSE = np.empty((3, len(NN_arch), len(n_grid)))
    MSE[:] = np.infty
    t = np.zeros((3, len(NN_arch), len(n_grid)))
    for index_n, n in enumerate(n_grid):
        print('Current n:', n)
        for index_arch, arch in enumerate(NN_arch):
            print('Current arch:', arch)
            t0 = time()
            model = NNPredict(
                verbose=0,
                es=True,
                es_give_up_after_nepochs=10,
                hidden_size=arch[1],
                num_layers=arch[0],
                gpu=False,
                scale_data=True,
                varying_theta0=True,
                fixed_theta0=False,
                dataloader_workers=0).fit(x_train[0:int(n*0.8)], y_train[0:int(n*0.8)])
            score = mse(model.predict(x_test[0:int(n/5)]), y_test[0:int(n/5)])
            if score < MSE[0, index_arch, index_n]:
                MSE[0, index_arch, index_n] = score
            t[0, index_arch, index_n] += time() - t0

            t0 = time()
            model = NNPredict(
                verbose=0,
                es=True,
                es_give_up_after_nepochs=10,
                hidden_size=arch[1],
                num_layers=arch[0],
                gpu=False,
                scale_data=True,
                varying_theta0=False,
                fixed_theta0=True,
                dataloader_workers=0).fit(x_train[0:int(n*0.8)], y_train[0:int(n*0.8)])
            score = mse(model.predict(x_test[0:int(n/5)]), y_test[0:int(n/5)])
            if score < MSE[1, index_arch, index_n]:
                MSE[1, index_arch, index_n] = score
            t[1, index_arch, index_n] += time() - t0

        for index_var, var in enumerate(LLE_var):
            print('Current var:', var)
            model = LLE().fit(x_train[0:int(n*0.8)], y_train[0:int(n*0.8)])
            score = mse(model.predict(x_test[0:int(n/5)], var), y_test[0:int(n/5)])
            if score < MSE[2, index_arch, index_n]:
                MSE[2, index_var, index_n] = score
            t[2, index_var, index_n] += time() - t0

    print(MSE, t)

    return MSE, t

output = apply(n_grid = [500, 1000, 2500, 5000, 10000, 20000, 50000, 100000], NN_arch = [[1, 100], [3, 300], [5, 500]], LLE_var = [0.1, 1, 10])

with open('Results/apply.pkl', 'wb') as file:
   	pickle.dump(output, file, pickle.HIGHEST_PROTOCOL)

