import numpy as np
from nnlocallinear import NLS, LLS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
from time import time
import pickle
import os
from copy import deepcopy
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True

x = np.linspace(0, 2*np.pi, num=2000)
y = np.sin(x)
x_train, x_test, y_train, y_test = train_test_split(x.reshape(2000, 1), y, test_size=0.2, random_state=0)
train = []
train_grad = []
test = []
test_grad = []
penal_grid = np.linspace(0.5, 2, num=4)

t0 = time()
model = NLS(
	verbose=2,
	es=True,
	es_give_up_after_nepochs=10,
	hidden_size=250,
	num_layers=3,
	gpu=False,
	scale_data=True,
	varying_theta0=False,
	fixed_theta0=True,
	penalization_thetas=0,
	dataloader_workers=0).fit(x_train, y_train)
pred, grad, trash = model.predict(x_train, pen_out = True)
train.append(mse(pred, y_train))
train_grad.append(grad.mean())
pred, grad, trash = model.predict(x_test, pen_out = True)
test.append(mse(pred, y_test))
test_grad.append(grad.mean())
pred0 = model.predict(x.reshape(2000, 1))
for penal in penal_grid:
	print('Current penalty', penal)
	t0 = time()
	model.penalization_thetas = penal
	model.improve_fit(x_train, y_train)
	pred, grad, trash = model.predict(x_train, pen_out = True)
	train.append(mse(pred, y_train))
	train_grad.append(grad.mean())
	pred, grad, trash = model.predict(x_test, pen_out = True)
	test.append(mse(pred, y_test))
	test_grad.append(grad.mean())
pred1 = model.predict(x.reshape(2000, 1))

penal_grid = [0] + penal_grid.tolist()
f = plt.figure()
topleft = plt.subplot2grid((2, 2), (0, 0))
topleft.plot(penal_grid, test, 'b-')
topleft.set(xlabel=r'$\lambda$', ylabel='MSE')
topright = plt.subplot2grid((2, 2), (0, 1))
topright.plot(penal_grid, test_grad, 'b-')
topright.set(xlabel=r'$\lambda$', ylabel='Average squared gradient')
bot = plt.subplot2grid((2, 2), (1, 0), colspan=2)
bot.plot(x, y, 'b-', label='True regression')
bot.plot(x, pred0, 'r-', label='NLS ($\lambda=0$)')
bot.plot(x, pred1, 'g-', label=r'NLS ($\lambda=2$)')
bot.set(xlabel='x', ylabel='y')
bot.legend()
f.tight_layout()
f.savefig('img/sin_toy_cvlambda2.pdf')




