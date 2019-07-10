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
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True

'''
amazon = pd.read_csv('/home/vcoscrato/Datasets/amazon fine foods.csv')[0:200000]
bow = CountVectorizer(stop_words='english', min_df=0.02, max_df=0.5)
x = bow.fit_transform(amazon['Text']).toarray()
features = np.array(bow.get_feature_names())
x = x[:, features != 'br']
features = features[features != 'br']
x_train, x_test, y_train, y_test = train_test_split(x, amazon['Score'], test_size=0.2, random_state=0)

def apply(n_grid, NN_arch, LLE_var):
    MSE = np.empty((3, len(n_grid)))
    MSE[:] = np.infty
    t = np.zeros((3, len(n_grid)))
    for index_n, n in enumerate(n_grid):
        print('Current n:', n)
        for arch in NN_arch:
            print('Current arch:', arch)
            t0 = time()
            model = NNPredict(
                verbose=0,
                es=True,
                es_give_up_after_nepochs=50,
                hidden_size=arch[1],
                num_layers=arch[0],
                gpu=False,
                scale_data=True,
                varying_theta0=True,
                fixed_theta0=False,
                dataloader_workers=0).fit(x_train[0:int(n*0.8)], y_train[0:int(n*0.8)])
            score = mse(model.predict(x_test[0:int(n/5)]), y_test[0:int(n/5)])
            if score < MSE[0, index_n]:
                MSE[0, index_n] = score
            t[0, index_n] += time() - t0

            t0 = time()
            model = NNPredict(
                verbose=0,
                es=True,
                es_give_up_after_nepochs=50,
                hidden_size=arch[1],
                num_layers=arch[0],
                gpu=False,
                scale_data=True,
                varying_theta0=False,
                fixed_theta0=True,
                dataloader_workers=0).fit(x_train[0:int(n*0.8)], y_train[0:int(n*0.8)])
            score = mse(model.predict(x_test[0:int(n/5)]), y_test[0:int(n/5)])
            if score < MSE[1, index_n]:
                MSE[1, index_n] = score
            t[1, index_n] += time() - t0

        for var in LLE_var:
            print('Current var:', var)
            t0 = time()
            model = LLE().fit(x_train[0:int(n*0.8)], y_train[0:int(n*0.8)])
            score = mse(model.predict(x_test[0:int(n/5)], var), y_test[0:int(n/5)])
            if score < MSE[2, index_n]:
                MSE[2, index_n] = score
            t[2, index_n] += time() - t0

    print(MSE, t)
    return MSE, t
'''

ngrid_ = [1000, 2500, 5000, 10000, 20000, 50000, 100000, 200000]
#output = apply(ngrid_, NN_arch = [[1, 100], [3, 100], [5, 100]], LLE_var = [0.1, 1, 10])

with open('results/LLE vs Fixed vs Varying.pkl', 'rb') as file:
   	output = pickle.load(file)

ngrid = [int(n/1000) for n in ngrid_]
#LLE vs Varying
f = plt.figure()
topleft = plt.subplot2grid((2, 3), (0, 0), colspan=2)
topleft.plot(output[0][0], 'r-', label=r'Varying $\theta_0$')
topleft.plot(output[0][2], 'b-', label=r'LLE')
topleft.set(xlabel=r'Sample size $(\times10^3)$', ylabel='MSE', xticks=range(len(ngrid)), xticklabels=ngrid)
topleft.legend()
botleft = plt.subplot2grid((2, 3), (1, 0), colspan=2)
botleft.plot(output[1][0], 'r-', label=r'Varying $\theta_0$')
botleft.plot(output[1][2], 'b-', label=r'LLE')
botleft.set(xlabel=r'Sample size $(\times10^3)$', ylabel='Fitting time', xticks=range(len(ngrid)), xticklabels=ngrid)
right = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
subset = range(3, len(ngrid))
right.plot(output[0][0, subset], 'r-', label=r'Varying $\theta_0$')
right.plot(output[0][2, subset], 'b-', label=r'LLE')
right.set(xlabel=r'Sample size $(\times10^3)$', ylabel='MSE', xticks=range(len([ngrid[i] for i in subset])), xticklabels=[ngrid[i] for i in subset])
f.tight_layout()
f.savefig('img/LLE vs Varying.pdf')

#Fixed vs Varying
f = plt.figure()
topleft = plt.subplot2grid((2, 3), (0, 0), colspan=2)
topleft.plot(output[0][0], 'r-', label=r'Varying $\theta_0$')
topleft.plot(output[0][1], 'g-', label=r'Fixed $\theta_0$')
topleft.set(xlabel=r'Sample size $(\times10^3)$', ylabel='MSE', xticks=range(len(ngrid)), xticklabels=ngrid)
topleft.legend()
botleft = plt.subplot2grid((2, 3), (1, 0), colspan=2)
botleft.plot(output[1][0], 'r-', label=r'Varying $\theta_0$')
botleft.plot(output[1][1], 'g-', label=r'Fixed $\theta_0$')
botleft.set(xlabel=r'Sample size $(\times10^3)$', ylabel='Fitting time', xticks=range(len(ngrid)), xticklabels=ngrid)
right = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
subset = range(3, len(ngrid))
right.plot(output[0][0, subset], 'r-', label=r'Varying $\theta_0$')
right.plot(output[0][1, subset], 'g-', label=r'Fixed $\theta_0$')
right.set(xlabel=r'Sample size $(\times10^3)$', ylabel='MSE', xticks=range(len([ngrid[i] for i in subset])), xticklabels=[ngrid[i] for i in subset])
f.tight_layout()
f.savefig('img/Fixed vs Varying.pdf')



