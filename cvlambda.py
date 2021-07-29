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
x = x.reshape(2000, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
test = []
test_grad = []
penal_grid = np.linspace(0.2, 1, num=5)

t0 = time()
model = NLS(
	verbose=2,
	es=True,
	es_give_up_after_nepochs=500,
	hidden_size=500,
	num_layers=3,
	gpu=False,
	scale_data=True,
	varying_theta0=False,
	fixed_theta0=True,
	penalization_thetas=0,
	dataloader_workers=0).fit(x_train, y_train)
pred, grad, trash = model.predict(x_test, grad_out=True)
test.append(mse(pred, y_test))
test_grad.append(grad.mean())
pred0 = model.predict(x)
theta0 = model.get_thetas(x, net_scale=False)[2]
for penal in penal_grid:
	print('Current penalty', penal)
	t0 = time()
	model.penalization_thetas = penal
	model.improve_fit(x_train, y_train)
	pred, grad, trash = model.predict(x_test, grad_out=True)
	test.append(mse(pred, y_test))
	test_grad.append(grad.mean())
pred1 = model.predict(x)
theta1 = model.get_thetas(x, net_scale=False)[2]

penal_grid = [0] + penal_grid.tolist()
f = plt.figure(figsize=(6,6))
topleft = plt.subplot2grid((3, 2), (0, 0))
topleft.plot(penal_grid, test, 'b-')
topleft.set(xlabel='$\lambda$', ylabel='MSE', xticks=penal_grid)
topright = plt.subplot2grid((3, 2), (0, 1), sharex=topleft)
topright.plot(penal_grid, test_grad, 'b-')
topright.set(xlabel=r'$\lambda$', ylabel='Average squared gradient')
mid = plt.subplot2grid((3, 2), (1, 0), colspan=2)
mid.plot(x, y, 'b-', label=r'y=sen(x)')
mid.plot(x, pred0, 'r-', label=r'NLS ($\lambda=0$)')
mid.plot(x, pred1, 'g-', label=r'NLS ($\lambda=1$)')
mid.set(xlabel='x', ylabel='y', xticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], xticklabels=['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
mid.legend()
bot = plt.subplot2grid((3, 2), (2, 0), colspan=2)
bot.plot(x, theta0, 'r-', label=r'NLS ($\lambda=0$)')
bot.plot(x, theta1, 'g-', label=r'NLS ($\lambda=1$)')
bot.set(xlabel='x', ylabel=r'$\beta_1(x)$', xticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], xticklabels=['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
bot.legend()
f.tight_layout()
f.savefig('img/sin_toy_cvlambda.pdf')




