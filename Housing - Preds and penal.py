import numpy as np
from nnlocallinear import NNPredict, LLE
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

data = pd.read_csv('/home/vcoscrato/Datasets/housing.csv')
data = data.sample(frac=1, random_state=0)
x = data.iloc[:, :13]
y = data.iloc[:, 13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=0)

def cvfit(NN_layers, NN_size, LLE_var):
	output = [[0, np.infty, [0, 0], 0], [0, np.infty, 0, 0]]
	for layers in NN_layers:
		print('Corrent layers:', layers)
		for size in NN_size:
			print('Current size:', size)
			t0 = time()
			model = NNPredict(
				verbose=0,
                es=True,
                es_give_up_after_nepochs=500,
                hidden_size=size,
                num_layers=layers,
                gpu=False,
                scale_data=True,
                varying_theta0=False,
                fixed_theta0=True,
                dataloader_workers=0).fit(x_train, y_train)
			score = mse(model.predict(x_test), y_test)
			if score < output[0][1]:
				output[0][0] = model
				output[0][1] = score
				output[0][2] = [layers, size]
			output[0][3] += time() - t0

	for var in LLE_var:
		print('Current var:', var)
		t0 = time()
		model = LLE().fit(x_train, y_train)
		score = mse(model.predict(x_test, var), y_test)
		if score < output[1][1]:
			output[1][0] = model
			output[1][1] = score
			output[1][2] = var
		output[1][3] += time() - t0

	return output

output = cvfit(NN_layers=[1, 2, 3, 5], NN_size=[100, 250, 500], LLE_var=[0.1, 0.5, 1, 5, 10, 50, 100])
with open('results/Housing_cvfit.txt', 'w') as f:
	print(output, file=f)


def cvlambda(penal_grid):
	output = np.zeros((len(penal_grid)+1, 3))
	print(output.shape)
	print('Current penalty', 0)
	t0 = time()
	model = NNPredict(
		verbose=0,
		es=True,
		es_give_up_after_nepochs=500,
		hidden_size=3,
		num_layers=250,
		gpu=False,
		scale_data=True,
		varying_theta0=False,
		fixed_theta0=True,
		penalization_thetas=0,
		dataloader_workers=0).fit(x_train, y_train)
	output[0, 0] = mse(model.predict(x_test), y_test)
	output[0, 2] = time() - t0
	for index, penal in enumerate(penal_grid):
		print('Current penalty', penal)
		t0 = time()
		model.penalization_thetas = penal
		model.improve_fit(x_train, y_train)
		output[index+1, 0] = mse(model.predict(x_test), y_test)
		output[index+1, 1] -= output[index+1, 0] + model.score(x_test, y_test)
		output[index+1, 2] = time() - t0

	return output

penal_grid = [x/10 for x in range(2, 40, 2)]
output = cvlambda(penal_grid)

penal_grid = [x/10 for x in range(0, 40, 2)]
f = plt.figure()
top = plt.subplot2grid((2, 2), (0, 0), colspan=2)
top.plot(penal_grid, output[:, 0])
top.set(xlabel=r'$\lambda$', ylabel='MSE')
botleft = plt.subplot2grid((2, 2), (1, 0))
botleft.plot(penal_grid, output[:, 1])
botleft.set(xlabel=r'$\lambda$', ylabel='Squared cumulate derivative')
botright = plt.subplot2grid((2, 2), (1, 1))
botright.plot(penal_grid, output[:, 2])
botright.set(xlabel=r'$\lambda$', ylabel='Adjust time')
f.tight_layout()
f.savefig('img/Boston_housing.pdf')


