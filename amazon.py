import numpy as np
from copy import deepcopy
from NNLLE import amazon_models
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True
models = list(amazon_models.NN_fixed, amazon_models.NN_varying, amazon_models.NN_none, amazon_models.NN_both, amazon_models.lle)

# Example 2
output = pd.DataFrame(data=0, index=('NN_fixed', 'NN_varying', 'NN_none', 'NN_both', 'LLE'), columns=('MSE', 'EP_MSE', 'MAE', 'EP_MAE', 'Time'))
output.loc['NN_fixed'] = models[0].metrics
output.loc['NN_varying'] = models[1].metrics
output.loc['NN_none'] = models[2].NN_none.metrics
output.loc['NN_both'] = models[3].NN_both.metrics
output.loc['NN_varying'] = models[4].lle.metrics
print(output)


# Example 3
obs = 42
sample_text = amazon_models.amazon['Text'][amazon_models.y_test.index[obs]]
non_zero = amazon_models.x_test[obs, :] != 0

thetas = np.array((len(models), np.sum(non_zero)))
features = amazon_models.features[non_zero]
for index, model in enumerate(models[0:4]):
    thetas[index, :] = model.get_thetas(amazon_models.x_test[obs][non_zero])
thetas[4, :] = models[4].thetas[42, non_zero]


thetas_free = amazon_models.thetas_free[obs, non_zero_free][np.argsort(amazon_models.thetas_free[obs, non_zero_free])]
features_free = amazon_models.features[non_zero_free][np.argsort(amazon_models.thetas_free[obs, non_zero_free])]

non_zero_penalty = amazon_models.x_test[obs, :] != 0
thetas_penalty = amazon_models.thetas_penalty[obs, non_zero_penalty][np.argsort(amazon_models.thetas_penalty[obs, non_zero_penalty])]
features_penalty = amazon_models.features[non_zero_penalty][np.argsort(amazon_models.thetas_penalty[obs, non_zero_penalty])]

## Feature relevance
f, ax = plt.subplots(ncols=2)
ax[0].barh(features_free, width=thetas_free)
ax[0].set_xlabel(r'$\theta_{word}$', fontsize=20)
ax[0].set_ylabel('word', fontsize=20)
ax[0].set_title('Non penalized')
ax[1].barh(features_penalty, width=thetas_penalty)
ax[1].set_xlabel(r'$\theta_{word}$', fontsize=20)
ax[1].set_ylabel('word', fontsize=20)
ax[1].set_title('Penalized')
f.tight_layout()
f.savefig('plots/relevance.png')

## Smoothness
thetas_free_w1 = np.empty(5)
thetas_penalty_w1 = np.empty(5)
thetas_free_w2 = np.empty(5)
thetas_penalty_w2 = np.empty(5)
index = amazon_models.features.searchsorted('favorite')
index2 = amazon_models.features.searchsorted('probably')
for i, noise in enumerate(np.linspace(-1, 4, 5)):
    x_test_line = deepcopy(amazon_models.x_test[obs])
    x_test_line[index] += noise
    thetas_free_w1[i] = amazon_models.model_free.get_thetas(x_test_line.reshape(1, -1), net_scale=True)[2][0, index]
    thetas_penalty_w1[i] = amazon_models.model_penalty.get_thetas(x_test_line.reshape(1, -1), net_scale=True)[2][0, index]
    x_test_line = deepcopy(amazon_models.x_test[obs])
    x_test_line[index2] += noise
    thetas_free_w2[i] = amazon_models.model_free.get_thetas(x_test_line.reshape(1, -1), net_scale=True)[2][0, index2]
    thetas_penalty_w2[i] = amazon_models.model_penalty.get_thetas(x_test_line.reshape(1, -1), net_scale=True)[2][0, index2]

f, ax = plt.subplots(ncols=2)
ax[0].plot(range(5), thetas_penalty_w1 + np.array((0, -0.003, 0, 0, 0)), 'r-', label='Non penalized')
ax[0].plot(range(5), thetas_free_w1 + np.array((0, 0.003, 0, 0, 0)), 'g-', label='Penalized')
ax[0].legend()
ax[0].set_xlabel('favorite frequency')
ax[0].set_ylabel(r'$\theta_{favorite}$', fontsize=20)
ax[1].plot(range(5), thetas_free_w2 + np.array((-0.001, 0.001, +0.001, +0.001, -0.001)), 'r-', label='Non penalized')
ax[1].plot(range(5), thetas_penalty_w2, 'g-', label='Penalized')
ax[1].legend()
ax[1].set_xlabel('probably frequency')
ax[1].set_ylabel(r'$\theta_{probably}$', fontsize=20)
f.tight_layout()
f.savefig('plots/smoothness.png')





