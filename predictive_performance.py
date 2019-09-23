import numpy as np
from nnlocallinear import NLS, LLS, NNPredict
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
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


def cvfit(x, y, NN_layers, NN_size, es_epochs, LLS_var, n_estimators):
	x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=0)
	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=0)
	output = []

	#NLS
	best_mse = np.infty
	t0 = time()
	print('NLS\'s...')
	for layers in NN_layers:
		print('Current layers:', layers)
		for size in NN_size:
			print('Current size:', size)
			model = NLS(
				verbose=0,
                es=True,
                es_give_up_after_nepochs=es_epochs,
                hidden_size=size,
                num_layers=layers,
                gpu=False,
                scale_data=True,
                varying_theta0=False,
                fixed_theta0=True,
                dataloader_workers=0).fit(x_train, y_train)
			pred = model.predict(x_val)
			mse = mean_squared_error(y_val, pred)
			if mse < best_mse:
			    best_mse = mse
				best_model = model
	best_pred = best_model.predict(x_test)
	best_mse = mean_squared_error(y_test, best_pred)
	mse_std = np.std((best_pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
	mae = mean_absolute_error(y_test, best_pred)
	mae_std = np.std(abs(best_pred.flatten()-y_test)) / (len(y_test)**(1/2))
	output.append([best_model, best_mse, mse_std, mae, mae_std, time()-t0])

	#NN
	best_mse = np.infty
	t0 = time()
	print('NN\'s...')
	for layers in NN_layers:
		print('Current layers:', layers)
		for size in NN_size:
			print('Current size:', size)
			x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size=1/10, random_state=0)
			model = NNPredict(
				verbose=0,
				es_give_up_after_nepochs=es_epochs,
				hidden_size=size,
				num_layers=layers,
				gpu=False).fit(x_train, y_train)
			pred = model.predict(x_val)
			mse = mean_squared_error(y_val, pred)
			if mse < best_mse:
			    best_mse = mse
				best_model = model
	best_pred = best_model.predict(x_test)
	best_mse = mean_squared_error(y_test, best_pred)
	mse_std = np.std((best_pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
	mae = mean_absolute_error(y_test, best_pred)
	mae_std = np.std(abs(best_pred.flatten()-y_test)) / (len(y_test)**(1/2))
	output.append([best_model, best_mse, mse_std, mae, mae_std, time()-t0])

	#LLS
	best_mse = np.infty
	t0 = time()
	print('LLS\'s...')
	for var in LLS_var:
		print('Current var:', var)
		model = LLS(kernel_var=var).fit(x_train, y_train)
		pred = model.predict(x_val)
		mse = mean_squared_error(y_val, pred)
		if mse < best_mse:
			best_mse = mse
			best_model = model
	best_pred = best_model.predict(x_test)
	best_mse = mean_squared_error(y_test, best_pred)
	mse_std = np.std((best_pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
	mae = mean_absolute_error(y_test, best_pred)
	mae_std = np.std(abs(best_pred.flatten()-y_test)) / (len(y_test)**(1/2))
	output.append([best_model, best_mse, mse_std, mae, mae_std, time()-t0])

	#RF
	best_mse = np.infty
	t0 = time()
	print('Random forests...')
	for n in n_estimators:
		print('Current n_estimators:', n)
		model = RandomForestRegressor(n_estimators=n).fit(x_train, y_train)
		pred = model.predict(x_val)
		mse = mean_squared_error(y_val, pred)
		if mse < best_mse:
			best_mse = mse
			best_model = model
	best_pred = best_model.predict(x_test)
	best_mse = mean_squared_error(y_test, best_pred)
	mse_std = np.std((best_pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
	mae = mean_absolute_error(y_test, best_pred)
	mae_std = np.std(abs(best_pred.flatten()-y_test)) / (len(y_test)**(1/2))
	output.append([best_model, best_mse, mse_std, mae, mae_std, time()-t0])

	return output

# Housing
print('Housing...')
data = pd.read_csv('/home/vcoscrato/Datasets/housing.csv')
data = data.sample(frac=1, random_state=0)
x = data.iloc[:, :13]
y = data.iloc[:, 13]

models = [NLS(verbose=0, es_give_up_after_nepochs=500, gpu=False, varying_theta0=False)]
parameters = [{'hidden_size':[100, 300], 'num_layers':[1, 3, 5]}]
models.append(NNPredict(verbose=0, es_give_up_after_nepochs=500, gpu=False))
parameters.append({'hidden_size':[100, 300], 'num_layers':[1, 3, 5]})
models.append(LLS())
parameters.append({'kernel_var':[0.1, 1, 10, 100, 1000]})
models.append(RandomForestRegressor())
parameters.append({'n_estimators':[10, 50, 100]})

output = [[], np.empty((4, 5))]
for index, (model, parameter) in enumerate(zip(models, parameters)):
	t0 = time()
	pred = cross_val_predict(GridSearchCV(model, parameter, scoring='neg_mean_squared_error', iid=False, 
		cv=KFold(n_splits=2, shuffle=False, random_state=0)), x, y, cv=KFold(n_splits=5, shuffle=False, random_state=0), n_jobs=-1)
	t = time() - t0
	mse = mean_squared_error(y, pred)
	mse_std = np.std((pred.flatten()-y)**2) / (len(y)**(1/2))
	mae = mean_absolute_error(y, pred)
	mae_std = np.std(abs(pred.flatten()-y)) / (len(y)**(1/2))
	output[0].append(pred)
	output[1][index, :] = [mse, mse_std, mae, mae_std, t]

with open('fitted/housing.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
with open('results/housing.txt', 'w') as f:
    print(output[1], file=f)
"""
# Superconductivity
print('Superconductivity...')
data = pd.read_csv('/home/vcoscrato/Datasets/superconductivity.csv')
data = data.sample(frac=1, random_state=0)
x = data.iloc[:, range(0, data.shape[1] - 1)].values
y = data.iloc[:, -1].values
output = cvfit(x, y, NN_layers=[1, 3, 5], NN_size=[100, 300], es_epochs=100, LLS_var=[0.1, 1, 10, 100, 1000], n_estimators=[10, 50, 100])
with open('fitted/superconductivity.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
with open('results/superconductivity.txt', 'w') as f:
    print(output, file=f)

# Blog
print('Blog...')
data = pd.read_csv('/home/vcoscrato/Datasets/blog feedback.csv')
x = data.iloc[:, range(0, data.shape[1] - 1)].values
y = data.iloc[:, -1].values
output = cvfit(x, y, NN_layers=[1, 3, 5], NN_size=[100, 300], es_epochs=100, LLS_var=[0.1, 1, 10, 100, 1000], n_estimators=[10, 50, 100])
with open('fitted/blog.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
with open('results/blog.txt', 'w') as f:
    print(output, file=f)

# Amazon
print('Amazon...')
amazon = pd.read_csv('/home/vcoscrato/Datasets/amazon fine foods.csv').sample(n=100000, random_state=0)
bow = CountVectorizer(stop_words='english', min_df=0.02, max_df=0.5)
x = bow.fit_transform(amazon['Text']).toarray()
features = np.array(bow.get_feature_names())
x = x[:, features != 'br']
features = features[features != 'br']
x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(x, amazon['Score'], test_size=0.2, shuffle=False, random_state=0)

n_grid = [1000, 2500, 5000, 10000, 25000, 50000, 100000]
output = []
for n in n_grid:
	print('Current n:', n)
	x_train = x_train_full[:int(n*0.8)]
	y_train = y_train_full[:int(n*0.8)]
	x_test = x_test_full[:int(n*0.2)]
	y_test = y_test_full[:int(n*0.2)]
	output.append(cvfit(NN_layers=[1, 3, 5], NN_size=[100, 300], es_epochs=100, LLS_var=[0.1, 1, 10, 100, 1000], n_estimators=[10, 50, 100]))

with open('fitted/amazon.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
with open('results/amazon.txt', 'w') as f:
    print(output, file=f)

n_grid = [1, 2.5, 5, 10, 25, 50, 100]
mse = np.empty((len(n_grid), 4))
time = np.empty((len(n_grid), 4))
for i in range(len(output)):
	for j in range(4):
		mse[i, j] = output[i][j][1]
		time[i, j] = output[i][j][5]

f = plt.figure()
topleft = plt.subplot2grid((2, 1), (0, 0))
topleft.plot(mse[:, 0], 'r-', label=r'NLS')
topleft.plot(mse[:, 1], 'g-', label=r'NN')
topleft.plot(mse[:, 2], 'b-', label=r'LLS')
topleft.plot(mse[:, 3], 'y-', label=r'RF')
topleft.set(xlabel=r'Sample size $(\times10^3)$', ylabel='MSE', xticks=range(len(n_grid)), xticklabels=n_grid)
topleft.legend()
botleft = plt.subplot2grid((2, 1), (1, 0), colspan=2)
botleft.plot(time[:, 0], 'r-', label=r'NLS')
botleft.plot(time[:, 1], 'g-', label=r'NN')
botleft.plot(time[:, 2], 'b-', label=r'LLS')
botleft.plot(time[:, 3], 'y-', label=r'RF')
botleft.set(xlabel=r'Sample size $(\times10^3)$', ylabel='Fitting time', xticks=range(len(n_grid)), xticklabels=n_grid)
f.tight_layout()
f.savefig('img/amazon_vary_n.pdf')
"""



