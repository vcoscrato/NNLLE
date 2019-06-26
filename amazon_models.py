from nnlocallinear import NNPredict, LLE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import pickle
import os

# Data
amazon = pd.read_csv('/home/vcoscrato/Datasets/amazon fine foods.csv')[0:30000]
bow = CountVectorizer(stop_words='english', min_df=0.02, max_df=0.5)
x = bow.fit_transform(amazon['Text']).toarray()
features = np.array(bow.get_feature_names())
x = x[:, features != 'br']
features = features[features != 'br']
x_train, x_test, y_train, y_test = train_test_split(x, amazon['Score'], test_size=0.2, random_state=0)


# Fixed
if os.path.isfile('models/NN_fixed.pkl'):
    with open('models/NN_fixed.pkl', 'rb') as file:
        NN_fixed = pickle.load(file)
else:
    t0 = time.time()
    NN_fixed = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=500,
        num_layers=5,
        gpu=False,
        scale_data=True,
        varying_theta0=False,
        fixed_theta0=True,
        dataloader_workers=0).fit(x_train, y_train)
    pred = NN_fixed.predict(x_test)
    NN_fixed.metrics = mse(pred, y_test), np.std((pred - np.array(y_test))**2)/(len(y_test)**(1/2)), mae(pred, y_test), np.std(abs(pred - np.array(y_test)))/(len(y_test)**(1/2)), time.time() - t0
    with open('models/NN_fixed.pkl', 'wb') as file:
    	pickle.dump(NN_fixed, file, pickle.HIGHEST_PROTOCOL)


# Varying
if os.path.isfile('models/NN_varying.pkl'):
    with open('models/NN_varying.pkl', 'rb') as file:
        NN_varying = pickle.load(file)
else:
    t0 = time.time()
    NN_varying = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=500,
        num_layers=5,
        gpu=False,
        scale_data=True,
        varying_theta0=True,
        fixed_theta0=False,
        dataloader_workers=0).fit(x_train, y_train)
    pred = NN_varying.predict(x_test)
    NN_varying.metrics = mse(pred, y_test), np.std((pred - np.array(y_test))**2)/(len(y_test)**(1/2)), mae(pred, y_test), np.std(abs(pred - np.array(y_test)))/(len(y_test)**(1/2)), time.time() - t0
    with open('models/NN_varying.pkl', 'wb') as file:
    	pickle.dump(NN_varying, file, pickle.HIGHEST_PROTOCOL)


# None
if os.path.isfile('models/NN_none.pkl'):
    with open('models/NN_none.pkl', 'rb') as file:
        NN_none = pickle.load(file)
else:
    t0 = time.time()
    NN_none = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=500,
        num_layers=5,
        gpu=False,
        scale_data=True,
        varying_theta0=False,
        fixed_theta0=False,
        dataloader_workers=0).fit(x_train, y_train)
    pred = NN_none.predict(x_test)
    NN_none.metrics = mse(pred, y_test), np.std((pred - np.array(y_test))**2)/(len(y_test)**(1/2)), mae(pred, y_test), np.std(abs(pred - np.array(y_test)))/(len(y_test)**(1/2)), time.time() - t0
    with open('models/NN_none.pkl', 'wb') as file:
    	pickle.dump(NN_none, file, pickle.HIGHEST_PROTOCOL)


# Both
if os.path.isfile('models/NN_both.pkl'):
    with open('models/NN_both.pkl', 'rb') as file:
        NN_both = pickle.load(file)
else:
    t0 = time.time()
    NN_both = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=500,
        num_layers=5,
        gpu=False,
        scale_data=True,
        varying_theta0=True,
        fixed_theta0=True,
        dataloader_workers=0).fit(x_train, y_train)
    pred = NN_both.predict(x_test)
    NN_both.metrics = mse(pred, y_test), np.std((pred - np.array(y_test))**2)/(len(y_test)**(1/2)), mae(pred, y_test), np.std(abs(pred - np.array(y_test)))/(len(y_test)**(1/2)), time.time() - t0
    with open('models/NN_both.pkl', 'wb') as file:
    	pickle.dump(NN_both, file, pickle.HIGHEST_PROTOCOL)


# LLE
if os.path.isfile('models/lle.pkl'):
    with open('models/lle.pkl', 'rb') as file:
        lle = pickle.load(file)
else:
    t0 = time.time()
    lle = LLE().fit(x_train, np.array(y_train).reshape(-1, 1))
    lle.preds, lle.thetas = lle.predict(x_test, 1, save_thetas=True)
    lle.metrics = mse(lle.preds, y_test), np.std((lle.preds - np.array(y_test))**2)/(len(y_test)**(1/2)), mae(lle.preds, y_test), np.std(abs(lle.preds - np.array(y_test)))/(len(y_test)**(1/2)), time.time() - t0
    with open('models/lle.pkl', 'wb') as file:
        pickle.dump(lle, file, pickle.HIGHEST_PROTOCOL)


