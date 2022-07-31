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

size = 100000
x = np.linspace(0, 2*np.pi, num=size)
y = np.sin(x)
x = x.reshape(size, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

penal_grid = [0, 1, 5, 10, 20]
MSE = []
avg_grad = []
preds = []
betas = []

model = NLS(
	nn_weight_decay=1e-6,
	verbose=0,
	es=True,
	es_give_up_after_nepochs=100,
	hidden_size=100,
	num_layers=3,
	gpu=True,
	scale_data=True,
	varying_theta0=False,
	fixed_theta0=True,
	penalization_thetas=0,
	optim_lr=1e-4,
	dataloader_workers=0)

for penal in penal_grid:
	print(model.actual_optim_lr)
	print('Current penalty', penal)
	model.penalization_thetas = penal
	if penal = 0:
		model.fit(x_train, y_train)
	else:
		model.improve_fit(x_train, y_train)
	pred, grad, _ = model.predict(x_test, grad_out=True)
	MSE.append(mse(pred, y_test))
	avg_grad.append(grad.mean())
	preds.append(model.predict(x))
	betas.append(model.get_thetas(x, net_scale=False)[2])

# Plots
f = plt.figure(figsize=(6, 6))
topleft = plt.subplot2grid((3, 2), (0, 0))
topleft.plot(penal_grid, MSE, 'b-')
topleft.set(xlabel='$\lambda$', ylabel='MSE', xticks=penal_grid)
topright = plt.subplot2grid((3, 2), (0, 1), sharex=topleft)
topright.plot(penal_grid, avg_grad, 'b-')
topright.set(xlabel=r'$\lambda$', ylabel='Average squared gradient')

mid = plt.subplot2grid((3, 2), (1, 0), colspan=2)
mid.plot(x, y, 'b-', label=r'y=sen(x)')
mid.plot(x, preds[0], 'r-', label=r'NLS ($\lambda=0$)')
mid.plot(x, preds[1], 'g-', label=r'NLS ($\lambda=1$)')
mid.plot(x, preds[2], 'y-', label=r'NLS ($\lambda=5$)')
mid.plot(x, preds[4], 'k-', label=r'NLS ($\lambda=20$)')
mid.set(xlabel='x', ylabel='y', xticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
        xticklabels=['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
mid.legend()

bot = plt.subplot2grid((3, 2), (2, 0), colspan=2)
bot.plot(x, betas[0], 'r-', label=r'NLS ($\lambda=0$)')
bot.plot(x, betas[1], 'g-', label=r'NLS ($\lambda=1$)')
bot.plot(x, betas[2], 'y-', label=r'NLS ($\lambda=5$)')
bot.plot(x, betas[4], 'k-', label=r'NLS ($\lambda=20$)')
bot.set(xlabel='x', ylabel=r'$\beta_1(x)$', xticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
        xticklabels=['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
bot.legend()
f.tight_layout()
f.savefig('img/sin_toy_cvlambda.pdf')
