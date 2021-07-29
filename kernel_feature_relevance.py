from nnlocallinear import NLS, LLS
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
matplotlib.rcParams['text.usetex'] = True

np.random.seed(0)
n = 2000
x = np.linspace(-5, 5, n).reshape(-1, 1)
y = x**2 + np.random.normal(0, 3, n).reshape(-1 ,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
scores = np.zeros((2, 3))
#indexes: (NLS or LLS, #irrelevant features, theta0 or theta1, instance)
thetas = np.empty((2, 3, 2, len(y)))

#0 features
nn = NLS(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=500,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
best = np.infty
for var in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]: 
    ll = LLS(var).fit(x_train, y_train.reshape(-1))
    if ll.score(x_test, y_test.reshape(-1)) < best:
        best = ll.score(x_test, y_test.reshape(-1))
        var_best = var
ll = LLS(var_best).fit(x_train, y_train.reshape(-1))
scores[:, 0] = [nn.score(x_test, y_test), ll.score(x_test, y_test.reshape(-1))]

#5 features
for i in range(0, 5):
    x2 = np.linspace(-5, 5, n).reshape(-1, 1)
    np.random.shuffle(x2)
    x = np.hstack((x, x2))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn = NLS(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=500,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
for var in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]: 
    ll = LLS(var).fit(x_train, y_train.reshape(-1))
    if ll.score(x_test, y_test.reshape(-1)) < best:
        best = ll.score(x_test, y_test.reshape(-1))
        var_best = var
ll = LLS(var_best).fit(x_train, y_train.reshape(-1))
scores[:, 1] = [nn.score(x_test, y_test), ll.score(x_test, y_test.reshape(-1))]

#50 features
for i in range(0, 45):
    x2 = np.linspace(-5, 5, n).reshape(-1, 1)
    np.random.shuffle(x2)
    x = np.hstack((x, x2))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn = NLS(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=500,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
for var in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]: 
    ll = LLS(var).fit(x_train, y_train.reshape(-1))
    if ll.score(x_test, y_test.reshape(-1)) < best:
        best = ll.score(x_test, y_test.reshape(-1))
        var_best = var
ll = LLS(var_best).fit(x_train, y_train.reshape(-1))
scores[:, 2] = [nn.score(x_test, y_test), ll.score(x_test, y_test.reshape(-1))]

#Output
np.savetxt('results/kernel feature relevance.txt', scores, delimiter=',')

f, ax = plt.subplots(figsize=(5, 3), ncols=2)
ax[0].plot(x[:, 0], y, 'ko', markersize=2)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('Relevant feature')
ax[1].plot(x[:, 1], y, 'ko', markersize=2)
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title('Irrelevant feature')
f.tight_layout()
f.savefig('img/relevant_irrelevant.pdf')