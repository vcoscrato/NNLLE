import numpy as np
from sklearn.model_selection import train_test_split
from nnlocallinear import NNPredict, LLE
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import pickle
import os

# Data
amazon = pd.read_csv('C:/Users/Victor/PycharmProjects/ML/NNLLE/Reviews.csv')[0:200000]
bow = CountVectorizer(stop_words='english', min_df=0.02)
x = bow.fit_transform(amazon['Text']).toarray()
features = np.array(bow.get_feature_names())
#x = x[:, features != 'br']
#features = features[features != 'br']
x_train, x_test, y_train, y_test = train_test_split(x, amazon['Score'], test_size=0.2, random_state=0)
output = pd.DataFrame(data=0, index=('NN0', 'NN100', 'LLE'), columns=('MSE', 'MAE', 'EP', 'Time'))

# Penalty
t0 = time.time()
print('a')
if os.path.isfile('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/NN_free.pkl'):
    with open('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/NN_free.pkl', 'rb') as file:
        model_free = pickle.load(file)
else:
    model_free = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=500,
        num_layers=5,
        gpu=False,
        tuningp=0,
        scale_data=False,
        varying_theta0=True,
        fixed_theta0=False,
        dataloader_workers=0).fit(x_train, y_train)
    with open('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/NN_free.pkl', 'wb') as file:
        pickle.dump(model_free, file, pickle.HIGHEST_PROTOCOL)
pred = model_free.predict(x_test).reshape(-1)
output.loc['NN0'] = (mse(pred, y_test), mae(pred, y_test),
                     np.std(np.array(y_test).reshape(-1,) - pred)/np.sqrt(len(y_test)), time.time() - t0)
thetas_free = model_free.get_thetas(x_test, net_scale=True)[2]

# Penalty
t0 = time.time()
print('a')
if os.path.isfile('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/NN_penalty.pkl'):
    with open('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/NN_penalty.pkl', 'rb') as file:
        model_penalty = pickle.load(file)
else:
    model_penalty = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=500,
        num_layers=5,
        gpu=False,
        tuningp=100,
        scale_data=False,
        varying_theta0=True,
        fixed_theta0=False,
        dataloader_workers=0).fit(x_train, y_train)
    with open('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/NN_penalty.pkl', 'wb') as file:
        pickle.dump(model_penalty, file, pickle.HIGHEST_PROTOCOL)
pred = model_penalty.predict(x_test).reshape(-1)
output.loc['NN100'] = (mse(pred, y_test), mae(pred, y_test),
                       np.std(np.array(y_test).reshape(-1,) - pred)/np.sqrt(len(y_test)), time.time() - t0)
thetas_penalty = model_penalty.get_thetas(x_test, net_scale=True)[2]

# LLE
x_test, y_test = x_test[0:500], y_test[0:500]
t0 = time.time()
print('a')
if os.path.isfile('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/LLE_preds.pkl'):
    with open('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/LLE_preds.pkl', 'rb') as file:
        pred = pickle.load(file)
    with open('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/LLE_thetas.pkl', 'rb') as file:
        thetasLLE = pickle.load(file)
else:
    model = LLE().fit(x_train, np.array(y_train).reshape(-1, 1))
    pred, thetas_LLE = model.predict(x_test, 1, save_thetas=True)
    with open('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/LLE_preds.pkl', 'wb') as file:
        pickle.dump(pred, file, pickle.HIGHEST_PROTOCOL)
    with open('C:/Users/Victor/PycharmProjects/ML/NNLLE/models/LLE_thetas.pkl', 'wb') as file:
        pickle.dump(thetas_LLE, file, pickle.HIGHEST_PROTOCOL)
pred = pred.reshape(-1)
output.loc['LLE'] = (mse(pred, y_test), mae(pred, y_test),
                     np.std(np.array(y_test).reshape(-1,) - pred)/np.sqrt(len(y_test)), time.time() - t0)
