from nnlocallinear import NNPredict, LLE
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
matplotlib.rcParams['text.usetex'] = True

np.random.seed(0)
x = np.linspace(-5, 5, 1000).reshape(-1, 1)
y = x**2 + np.random.normal(0, 3, 1000).reshape(-1 ,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
scores = np.zeros((2, 3))
#indexes: (nnlle or lle, #irrelevant features, theta0 or theta1, instance)
thetas = np.empty((2, 3, 2, len(y)))

#0 features
nn = NNPredict(
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
ll = LLE().fit(x_train, y_train.reshape(-1))


scores[:, 0] = [nn.score(x_test, y_test), ll.score(x_test, y_test.reshape(-1), 1)]
thetas_nn = nn.get_thetas(x, net_scale=False)
thetas[0, 0, 0, :] = thetas_nn[0][:, 0]
thetas[0, 0, 1, :] = thetas_nn[2][:, 0]
thetas_ll = ll.predict(x, 1, save_thetas=True)[1]
thetas[1, 0, 0, :] = thetas_ll[:, 0]
thetas[1, 0, 1, :] = thetas_ll[:, 1]

#5 features
for i in range(0, 5):
    x2 = np.linspace(-10, 10, 1000).reshape(-1, 1)
    np.random.shuffle(x2)
    x = np.hstack((x, x2))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn = NNPredict(
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
ll = LLE().fit(x_train, y_train.reshape(-1))

scores[:, 1] = [nn.score(x_test, y_test), ll.score(x_test, y_test.reshape(-1), 1)]
thetas_nn = nn.get_thetas(x, net_scale=False)
thetas[0, 1, 0, :] = thetas_nn[0][:, 0]
thetas[0, 1, 1, :] = thetas_nn[2][:, 0]
thetas_ll = ll.predict(x, 1, save_thetas=True)[1]
thetas[1, 1, 0, :] = thetas_ll[:, 0]
thetas[1, 1, 1, :] = thetas_ll[:, 1]

#50 features
for i in range(0, 45):
    x2 = np.linspace(-10, 10, 1000).reshape(-1, 1)
    np.random.shuffle(x2)
    x = np.hstack((x, x2))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn = NNPredict(
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
ll = LLE().fit(x_train, y_train.reshape(-1))

scores[:, 2] = [nn.score(x_test, y_test), ll.score(x_test, y_test.reshape(-1), 1)]
thetas_nn = nn.get_thetas(x, net_scale=False)
thetas[0, 2, 0, :] = thetas_nn[0][:, 0]
thetas[0, 2, 1, :] = thetas_nn[2][:, 0]
thetas_ll = ll.predict(x, 1, save_thetas=True)[1]
thetas[1, 2, 0, :] = thetas_ll[:, 0]
thetas[1, 2, 1, :] = thetas_ll[:, 1]

#plot
print(x.shape, thetas.shape, thetas[0, 0, 0, :].shape)
f, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].plot(x[:, 1], thetas[0, 0, 0, :], 'ro', label='No features', linewidth=0.2)
ax[0, 0].plot(x[:, 1], thetas[0, 1, 0, :], 'go', label='5 features', linewidth=0.2)
ax[0, 0].plot(x[:, 1], thetas[0, 2, 0, :], 'bo', label='50 features', linewidth=0.2)
ax[0, 0].legend()
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel(r'$\theta_0(x)$')
ax[0, 0].set_title(r'$\theta_0(x)$ on NNLLE')
ax[0, 1].plot(x[:, 1], thetas[0, 0, 1, :], 'ro', label='No features', linewidth=0.2)
ax[0, 1].plot(x[:, 1], thetas[0, 1, 1, :], 'go', label='5 features', linewidth=0.2)
ax[0, 1].plot(x[:, 1], thetas[0, 2, 1, :], 'bo', label='50 features', linewidth=0.2)
ax[0, 1].legend()
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel(r'$\theta_1(x)$')
ax[0, 1].set_title(r'$\theta_1(x)$ on NNLLE')
ax[0, 1].legend()
ax[1, 0].plot(x[:, 1], thetas[1, 0, 0, :], 'ro', label='No features', linewidth=0.2)
ax[1, 0].plot(x[:, 1], thetas[1, 1, 0, :], 'go', label='5 features', linewidth=0.2)
ax[1, 0].plot(x[:, 1], thetas[1, 2, 0, :], 'bo', label='50 features', linewidth=0.2)
ax[1, 0].legend()
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel(r'$\theta_0(x)$')
ax[1, 0].set_title(r'$\theta_0(x)$ on LLE')
ax[1, 1].plot(x[:, 1], thetas[1, 0, 1, :], 'ro', label='No features', linewidth=0.2)
ax[1, 1].plot(x[:, 1], thetas[1, 1, 1, :], 'go', label='5 features', linewidth=0.2)
ax[1, 1].plot(x[:, 1], thetas[1, 2, 1, :], 'bo', label='50 features', linewidth=0.2)
ax[1, 1].legend()
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel(r'$\theta_1(x)$')
ax[1, 1].set_title(r'$\theta_1(x)$ on LLE')
ax[1, 1].legend()
f.tight_layout()
f.savefig('img/thetas kernel feature relevance.pdf')

np.savetxt('results/kernel feature relevance.txt', scores, delimiter=',')

