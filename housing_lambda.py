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
model = NLS(verbose=0, es_give_up_after_nepochs=500, gpu=False, varying_theta0=False, num_layers=3, hidden_size=3)

# Split predictions given and samples to extend predictions for
x_pred, x_extend, y_pred, y_extend = train_test_split(x_test, y_test, test_size=0.25, shuffle=False, random_state=0)
dists = euclidean_distances(x_extend, x_pred)

# Set up penalization grid
lambda_grid = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 9999999999999]
output = np.empty((len(lambda_grid), 5))
true_preds = np.empty((len(lambda_grid), len(x_extend)))
extended_preds = np.empty((len(lambda_grid), len(x_extend)))

# Calculate models and metrics
for index, penalty in enumerate(lambda_grid):
    print('Current lambda:', penalty)
    t0 = time()
    model.penalization_thetas=penalty
    if index == 0:
        model.fit(x_train, y_train)
    else:
        model.improve_fit(x_train, y_train)
    pred, grad, trash = model.predict(x_test, grad_out = True)
    pred2, grad2, trash = model.predict(x_train, grad_out = True)
    output[index, :] = [mean_squared_error(y_test, pred), grad.mean(), mean_squared_error(y_train, pred2), grad2.mean(), time()-t0]
    true_preds[index] = model.predict(x_extend).reshape(-1)
    for i in range(len(x_extend)):
        neighbord_index = np.argmin(dists[i])
        neighbord_thetas = model.get_thetas(x_pred[neighbord_index].reshape(1, -1), net_scale=False)
        extended_preds[index, i] = neighbord_thetas[0] + neighbord_thetas[1] + np.inner(neighbord_thetas[2], x_extend[i])
errors = np.abs(true_preds - extended_preds).mean(axis=1)
np.savetxt('results/housing_lambda.txt', output, delimiter=',')
np.savetxt('results/errors_.txt', errors, delimiter=',')

# Output results
lambda_grid = [str(n) for n in lambda_grid[:-1]] + [r'$\infty$']
f = plt.figure()
top = plt.subplot2grid((2, 1), (0, 0))
top.plot(output[:, 0], 'b-', label = 'Test data')
top.plot(output[:, 2], 'y-', label = 'Train data')
top.set(xlabel='$\lambda$', ylabel='MSE', xticks=range(len(lambda_grid)), xticklabels=lambda_grid)
top.legend()
botleft = plt.subplot2grid((2, 2), (1, 0), sharex=top)
botleft.plot(output[:, 1], 'b-', label = 'Test data')
botleft.plot(output[:, 3], 'y-', label = 'Train data')
botleft.set(xlabel=r'$\lambda$', ylabel='Average squared gradient')
botleft.legend()
botright = plt.subplot2grid((2, 2), (1, 1))
botright.plot(output[3:, 1], 'b-', label = 'Test data')
botright.plot(output[3:, 3], 'y-', label = 'Train data')
botright.set(xlabel='$\lambda$', ylabel='MSE', xticks=range(len(lambda_grid))[:-3], xticklabels=lambda_grid[3:])
botright.set(xlabel=r'$\lambda$', ylabel='Average squared gradient')
botright.legend()
f.tight_layout()
f.savefig('img/housing_cvlambda.pdf')

f = plt.figure()
left = plt.subplot2grid((2, 1), (0, 0))
left.plot(lambda_grid, errors, 'b-')
left.set(xlabel='$\lambda$', ylabel='Average extension error', xticks=range(len(lambda_grid)), xticklabels=lambda_grid)
right = plt.subplot2grid((2, 1), (1, 0))
right.plot(lambda_grid[3:], errors[3:], 'b-')
right.set(xlabel='$\lambda$', ylabel='Average extension error', xticks=range(len(lambda_grid))[:-3], xticklabels=lambda_grid[3:])
f.tight_layout()
f.savefig('img/housing_extend.pdf')







