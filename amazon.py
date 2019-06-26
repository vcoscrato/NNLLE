import numpy as np
import pandas as pd
import amazon_models
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True
models = [amazon_models.NN_fixed, amazon_models.NN_varying, amazon_models.NN_none, amazon_models.NN_both, amazon_models.lle]

# Example 2
output = pd.DataFrame(data=0, index=('NN_fixed', 'NN_varying', 'NN_none', 'NN_both', 'LLE'), columns=('MSE', 'EP_MSE', 'MAE', 'EP_MAE', 'Time'))
output.loc['NN_fixed'] = models[0].metrics
output.loc['NN_varying'] = models[1].metrics
output.loc['NN_none'] = models[2].metrics
output.loc['NN_both'] = models[3].metrics
output.loc['LLE'] = models[4].metrics
print(output)


# Example 3
obs = 42
sample_text = amazon_models.amazon['Text'][amazon_models.y_test.index[obs]]
non_zero = amazon_models.x_test[obs, :] != 0

thetas = np.empty((len(models), np.sum(non_zero)))
features = amazon_models.features[non_zero]
index_top = list()

for index, model in enumerate(models[0:4]):
    thetas[index, :] = model.get_thetas(amazon_models.x_test[obs].reshape(1, -1), net_scale=False)[2].reshape(-1)[non_zero]
    index_top.append(np.argsort(np.abs(thetas[index, :]))[:10])
thetas[4, :] = models[4].thetas[42, 1:][non_zero]
index_top.append(np.argsort(np.abs(thetas[4, :]))[:10])

f, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].barh(features[index_top[0]], width=thetas[0, index_top[0]])
ax[0, 0].set_title('Fixed')
ax[0, 1].barh(features[index_top[1]], width=thetas[1, index_top[1]])
ax[0, 1].set_title('Varying')
ax[1, 0].barh(features[index_top[2]], width=thetas[2, index_top[2]])
ax[1, 0].set_title('None')
ax[1, 1].barh(features[index_top[3]], width=thetas[3, index_top[3]])
ax[1, 1].set_title('Both')
for axes in ax.ravel():
    axes.set_xlabel(r'$theta_{word}$')
    axes.set_ylabel(r'$word$')
f.tight_layout()
f.savefig('img/choose theta0.pdf')

f, ax = plt.subplots()
ax.barh(features[index_top[4]], width=thetas[4, index_top[4]])
ax.set_title('LLE')
ax.set_xlabel(r'$theta_{word}$')
ax.set_ylabel(r'$word$')
f.tight_layout()
f.savefig('img/lle relevance.pdf')




