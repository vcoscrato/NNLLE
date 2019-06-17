import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from nnlocallinear import NNPredict, LLE

np.random.seed(0)
scores = np.zeros((2, 3))

x = np.linspace(-10, 10, 2000).reshape(-1, 1)
y = x**2 + np.random.normal(0, 3, 2000).reshape(-1 ,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn1 = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    tuningp=0,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
lle1 = LLE().fit(x_train, y_train)
scores[:, 0] = [nn1.score(x_test, y_test), lle1.score(x_test, y_test, 1)]

for i in range(0, 5):
    x2 = np.linspace(-10, 10, 2000).reshape(-1, 1)
    np.random.shuffle(x2)
    x = np.hstack((x, x2))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn2 = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    tuningp=0,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
lle2 = LLE().fit(x_train, y_train)
scores[:, 1] = [nn2.score(x_test, y_test), lle2.score(x_test, y_test, 1)]

for i in range(0, 45):
    x2 = np.linspace(-10, 10, 2000).reshape(-1, 1)
    np.random.shuffle(x2)
    x = np.hstack((x, x2))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
nn3 = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=3,
    gpu=False,
    tuningp=0,
    scale_data=True,
    varying_theta0=True,
    fixed_theta0=False,
    dataloader_workers=0).fit(x_train, y_train)
lle3 = LLE().fit(x_train, y_train)
scores[:, 2] = [nn3.score(x_test, y_test), lle3.score(x_test, y_test, 1)]

f, ax = plt.subplots(ncols=2)
ax[0].plot(x[:, 0], y, 'ko')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('Relevant feature')
ax[1].plot(x[:, 1], y, 'ko')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title('Irrelevant feature')
f.tight_layout()
f.savefig('plots/simulation.pdf')

print(scores)


