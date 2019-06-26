import numpy as np
import pickle
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True


with open('Results/apply.pkl', 'rb') as file:
   	output = pickle.load(file)

best_MSE = np.max(output[0][:, :, 3:6], axis = 1)
times = np.sum(output[1][:, :, 3:6], axis = 1)

f, ax = plt.subplots(nrows = 2)
ax[0].plot(best_MSE[0], 'r-', label=r'Varying $\theta_0$')
ax[0].plot(best_MSE[1], 'g-', label=r'Fixed $\theta_0$')
ax[0].plot(best_MSE[2], 'b-', label=r'LLE')
ax[0].legend()
ax[0].set_xlabel('Sample size')
ax[0].set_xticks(range(3))
ax[0].set_xticklabels([5000, 10000, 20000])
ax[0].set_ylabel('MSE')
ax[1].plot(times[0], 'r-', label=r'Varying $\theta_0$')
ax[1].plot(times[1], 'g-', label=r'Fixed $\theta_0$')
ax[1].plot(times[2], 'b-', label=r'LLE')
ax[1].legend()
ax[1].set_xlabel('Sample size')
ax[1].set_xticks(range(3))
ax[1].set_xticklabels([5000, 10000, 20000])
ax[1].set_ylabel('Fitting time')
f.tight_layout()
f.savefig('img/amazon.pdf')
