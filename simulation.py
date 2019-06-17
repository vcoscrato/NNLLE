from nnlocallinear import NNPredict, LLE
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
matplotlib.rcParams['text.usetex'] = True

np.random.seed(0)
scores = np.zeros((3, 3))

x = np.linspace(-10, 10, 2000).reshape(-1, 1)
y = x**2 + np.random.normal(0, 3, 2000).reshape(-1 ,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

nn_varying = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
nn_fixed = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=False,
    fixed_theta0=True,
    dataloader_workers=0).fit(x_train, y_train)
lle = LLE().fit(x_train, y_train)

scores[:, 0] = [nn_varying.score(x_test, y_test), nn_fixed.score(x_test, y_test), lle.score(x_test, y_test, 1)]
thetas_vary = nn_varying.get_thetas(x, net_scale=False)
thetas_fixed = nn_fixed.get_thetas(x, net_scale=False)
thetas_lle = lle.predict(x, 1, save_thetas=True)

f, ax = plt.subplots(ncols=3)
ax[0].plot(y, nn_varying.predict(x), 'r-', label='NNLLE varying', linewidth=0.2)
ax[0].plot(y, nn_fixed.predict(x), 'g-', label='NNLLE fixed', linewidth=0.2)
ax[0].plot(y, thetas_lle[0], 'b-', label='LLE', linewidth=0.2)
ax[0].set_xlabel(r'$y_{true}$', fontsize=20)
ax[0].set_ylabel(r'$y_{pred}$', fontsize=20)
ax[0].legend()
ax[1].plot(x, thetas_vary[0], 'ro', label='NNLLE varying')
ax[1].plot(x, np.repeat(thetas_fixed[1], len(x)), 'go', label='NNLLE fixed')
ax[1].plot(x, thetas_lle[1][:, 0], 'bo', label='LLE')
ax[1].set_xlabel('x', fontsize=20)
ax[1].set_ylabel(r'$\theta_0$', fontsize=20)
ax[1].legend()
ax[2].plot(x, thetas_vary[2], 'ro', label='NNLLE varying')
ax[2].plot(x, thetas_fixed[2], 'go', label='NNLLE fixed')
ax[2].plot(x, thetas_lle[1][:, 1], 'bo', label='LLE')
ax[2].set_xlabel('x', fontsize=20)
ax[2].set_ylabel(r'$\theta_1$', fontsize=20)
ax[2].legend()
f.tight_layout()
f.savefig('img/fixed_vs_varying0.pdf')


for i in range(0, 5):
    x2 = np.linspace(-10, 10, 2000).reshape(-1, 1)
    np.random.shuffle(x2)
    x = np.hstack((x, x2))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn_varying = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
nn_fixed = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=False,
    fixed_theta0=True,
    dataloader_workers=0).fit(x_train, y_train)
lle = LLE().fit(x_train, y_train)

scores[:, 1] = [nn_varying.score(x_test, y_test), nn_fixed.score(x_test, y_test), lle.score(x_test, y_test, 1)]
thetas_vary = nn_varying.get_thetas(x, net_scale=False)
thetas_fixed = nn_fixed.get_thetas(x, net_scale=False)
thetas_lle = lle.predict(x, 1, save_thetas=True)

f, ax = plt.subplots(ncols=3)
ax[0].plot(y, nn_varying.predict(x), 'r-', label='NNLLE varying', linewidth=0.2)
ax[0].plot(y, nn_fixed.predict(x), 'g-', label='NNLLE fixed', linewidth=0.2)
ax[0].set_xlabel(r'$y_{true}$', fontsize=20)
ax[0].set_ylabel(r'$y_{pred}$', fontsize=20)
ax[0].legend()
ax[1].plot(x[:, 0], thetas_vary[0], 'ro', label='NNLLE varying', markersize=0.5)
ax[1].plot(x[:, 0], np.repeat(thetas_fixed[1], len(x)), 'go', label='NNLLE fixed', markersize=0.5)
ax[1].set_xlabel('x', fontsize=20)
ax[1].set_ylabel(r'$\theta_0$', fontsize=20)
ax[1].legend()
ax[2].plot(x[:, 0], thetas_vary[2][:, 0], 'ro', label='NNLLE varying', markersize=0.5)
ax[2].plot(x[:, 0], thetas_fixed[2][:, 0], 'go', label='NNLLE fixed', markersize=0.5)
ax[2].set_xlabel('x', fontsize=20)
ax[2].set_ylabel(r'$\theta_1$', fontsize=20)
ax[2].legend()
f.tight_layout()
f.savefig('img/fixed_vs_varying5.pdf')

for i in range(0, 45):
    x2 = np.linspace(-10, 10, 2000).reshape(-1, 1)
    np.random.shuffle(x2)
    x = np.hstack((x, x2))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn_varying = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
nn_fixed = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    scale_data=True,
    varying_theta0=False,
    fixed_theta0=True,
    dataloader_workers=0).fit(x_train, y_train)
lle = LLE().fit(x_train, y_train)

scores[:, 2] = [nn_varying.score(x_test, y_test), nn_fixed.score(x_test, y_test), lle.score(x_test, y_test, 1)]
thetas_vary = nn_varying.get_thetas(x, net_scale=False)
thetas_fixed = nn_fixed.get_thetas(x, net_scale=False)
thetas_lle = lle.predict(x, 1, save_thetas=True)

f, ax = plt.subplots(ncols=3)
ax[0].plot(y, nn_varying.predict(x), 'r-', label='NNLLE varying', linewidth=0.2)
ax[0].plot(y, nn_fixed.predict(x), 'g-', label='NNLLE fixed', linewidth=0.2)
ax[0].set_xlabel(r'$y_{true}$', fontsize=20)
ax[0].set_ylabel(r'$y_{pred}$', fontsize=20)
ax[0].legend()
ax[1].plot(x[:, 0], thetas_vary[0], 'ro', label='NNLLE varying', markersize=0.5)
ax[1].plot(x[:, 0], np.repeat(thetas_fixed[1], len(x)), 'go', label='NNLLE fixed', markersize=0.5)
ax[1].set_xlabel('x', fontsize=20)
ax[1].set_ylabel(r'$\theta_0$', fontsize=20)
ax[1].legend()
ax[2].plot(x[:, 0], thetas_vary[2][:, 0], 'ro', label='NNLLE varying', markersize=0.5)
ax[2].plot(x[:, 0], thetas_fixed[2][:, 0], 'go', label='NNLLE fixed', markersize=0.5)
ax[2].set_xlabel('x', fontsize=20)
ax[2].set_ylabel(r'$\theta_1$', fontsize=20)
ax[2].legend()
f.tight_layout()
f.savefig('img/fixed_vs_varying50.pdf')

print(scores)


