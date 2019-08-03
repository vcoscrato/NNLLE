import numpy as np
from nnlocallinear import NLS, LLS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from time import time
import pickle
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True

# Load data
data = pd.read_csv('/home/vcoscrato/Datasets/housing.csv')
data = data.sample(frac=1, random_state=0).values
x = data[:, :13]
y = data[:, 13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=0)
with open('fitted/housing.pkl', 'rb') as f:
	model = pickle.load(f)[0]
t = model[5]
model = model[0]

# Split predictions given and samples to extend predictions for
x_pred, x_extend, y_pred, y_extend = train_test_split(x_test, y_test, test_size=0.25, shuffle=False, random_state=0)
dists = euclidean_distances(x_extend, x_pred)

# Set up penalization grid
lambda_grid = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 9999999999999]
output = np.empty((len(lambda_grid)+1, 5))
true_preds = np.empty((len(lambda_grid)+1, len(x_extend)))
extended_preds = np.empty((len(lambda_grid)+1, len(x_extend)))

# Calculate models and metrics
pred, grad, trash = model.predict(x_test, grad_out = True)
pred2, grad2, trash = model.predict(x_train, grad_out = True)
output[0, :] = [mean_squared_error(y_test, pred), grad.mean(), mean_squared_error(y_train, pred2), grad2.mean(), t/9]
true_preds[0] = model.predict(x_extend).reshape(-1)
for i in range(len(x_extend)):
    neighbord_index = np.argmin(dists[i])
    neighbord_thetas = model.get_thetas(x_pred[neighbord_index].reshape(1, -1), net_scale=False)
    extended_preds[0, i] = neighbord_thetas[0] + neighbord_thetas[1] + np.inner(neighbord_thetas[2], x_extend[i])
for index, penalty in enumerate(lambda_grid):
    print('Current lambda:', penalty)
    t0 = time()
    model.penalization_thetas=penalty
    model.improve_fit(x_train, y_train)
    pred, grad, trash = model.predict(x_test, grad_out = True)
    pred2, grad2, trash = model.predict(x_train, grad_out = True)
    output[index+1, :] = [mean_squared_error(y_test, pred), grad.mean(), mean_squared_error(y_train, pred2), grad2.mean(), time()-t0]
    true_preds[index+1] = model.predict(x_extend).reshape(-1)
    for i in range(len(x_extend)):
        neighbord_index = np.argmin(dists[i])
        neighbord_thetas = model.get_thetas(x_pred[neighbord_index].reshape(1, -1), net_scale=False)
        extended_preds[index+1, i] = neighbord_thetas[0] + neighbord_thetas[1] + np.inner(neighbord_thetas[2], x_extend[i])
errors = np.abs(true_preds - extended_preds).mean(axis=1)
np.savetxt('results/housing_lambda.txt', output, delimiter=',')
np.savetxt('results/errors_.txt', errors, delimiter=',')

# Output results
lambda_grid = ['0'] + [str(n) for n in lambda_grid[:-1]] + [r'$\infty$']
f = plt.figure()
top = plt.subplot2grid((2, 1), (0, 0))
top.plot(output[:, 0], 'b-', label = 'Test data')
top.plot(output[:, 2], 'y-', label = 'Train data')
top.set(xlabel='$\lambda$', ylabel='MSE', xticks=range(len(lambda_grid)), xticklabels=lambda_grid)
top.legend()
bot = plt.subplot2grid((2, 1), (1, 0), sharex=top)
bot.plot(output[:, 1], 'b-', label = 'Test data')
bot.plot(output[:, 3], 'y-', label = 'Train data')
bot.set(xlabel=r'$\lambda$', ylabel='Average squared gradient')
bot.legend()
f.tight_layout()
f.savefig('img/housing_cvlambda.pdf')

f, ax = plt.subplots()
f.figsize = [3.2, 4.8]
ax.plot(lambda_grid, errors, 'b-')
ax.set(xlabel='$\lambda$', ylabel='Average extension error', xticks=range(len(lambda_grid)), xticklabels=lambda_grid)
f.tight_layout()
f.savefig('img/housing_extend.pdf')







