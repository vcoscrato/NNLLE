import numpy as np
from nnlocallinear import NNPredict, LLE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from time import time
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True


def cvfit(NN_layers, NN_size, es_epochs, LLE_var, n_estimators):
	output = []

	#NNLLE
	best_mse = np.infty
	t0 = time()
	print('NNLLEs...')
	for layers in NN_layers:
		print('Current layers:', layers)
		for size in NN_size:
			print('Current size:', size)
			model = NNPredict(
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
			pred = model.predict(x_test)
			mse = mean_squared_error(y_test, pred)
			if mse < best_mse:
				best_mse = mse
				best_model = model
				best_pred = pred
	mse_std = np.std((best_pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
	mae = mean_absolute_error(y_test, best_pred)
	mae_std = np.std(abs(best_pred.flatten()-y_test)) / (len(y_test)**(1/2))
	output.append([best_model, best_mse, mse_std, mae, mae_std, time()-t0])

	#NN
	best_mse = np.infty
	t0 = time()
	print('NNs...')
	for layers in NN_layers:
		print('Current layers:', layers)
		for size in NN_size:
			print('Current size:', size)
			x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size=1/10, random_state=0)
			model = Sequential()
			for i in range(layers):
				model.add(Dense(size, input_shape=(x.shape[1],), activation='elu'))
				model.add(Dropout(0.5))
			model.add(Dense(1, activation='linear'))
			model.compile(loss='mse', optimizer='adam')
			callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=es_epochs, verbose=0, mode='auto')]
			model.fit(x_train1, y_train1, epochs=10000, batch_size=256, verbose=0, callbacks=callbacks, validation_data=(x_val, y_val))
			pred = model.predict(x_test)
			mse = mean_squared_error(y_test, pred)
			if mse < best_mse:
				best_mse = mse
				best_model = model
				best_pred = pred
	mse_std = np.std((best_pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
	mae = mean_absolute_error(y_test, best_pred)
	mae_std = np.std(abs(best_pred.flatten()-y_test)) / (len(y_test)**(1/2))
	output.append([best_model, best_mse, mse_std, mae, mae_std, time()-t0])

	#LLE
	best_mse = np.infty
	t0 = time()
	print('LLEs...')
	for var in LLE_var:
		print('Current var:', var)
		model = LLE(kernel_var=var).fit(x_train, y_train)
		pred = model.predict(x_test)
		mse = mean_squared_error(y_test, pred)
		if mse < best_mse:
			best_mse = mse
			best_model = model
			best_pred = pred
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
		pred = model.predict(x_test)
		mse = mean_squared_error(y_test, pred)
		if mse < best_mse:
			best_mse = mse
			best_model = model
			best_pred = pred
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=0)
output = cvfit(NN_layers=[1, 3, 5], NN_size=[100, 250, 500], es_epochs=500, LLE_var=[0.1, 1, 10, 100, 1000], n_estimators=[10, 50, 100])
with open('results/housing.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

# Superconductivity
print('Superconductivity...')
data = pd.read_csv('/home/vcoscrato/Datasets/superconductivity.csv')
data = data.sample(frac=1, random_state=0)
x = data.iloc[:, range(0, data.shape[1] - 1)].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=0)
output = cvfit(NN_layers=[1, 3, 5], NN_size=[100, 250, 500], es_epochs=100, LLE_var=[0.1, 1, 10, 100, 1000], n_estimators=[10, 50, 100])
with open('results/superconductivity.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

# Blog
print('Blog...')
data = pd.read_csv('/home/vcoscrato/Datasets/blog feedback.csv')
x = data.iloc[:, range(0, data.shape[1] - 1)].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=0)
output = cvfit(NN_layers=[1, 3, 5], NN_size=[100, 250, 500], es_epochs=100, LLE_var=[0.1, 1, 10, 100, 1000], n_estimators=[10, 50, 100])
with open('results/blog_feedback.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

# Amazon
print('Amazon...')
amazon = pd.read_csv('/home/vcoscrato/Datasets/amazon fine foods.csv').sample(n=100000, random_state=0)
bow = CountVectorizer(stop_words='english', min_df=0.02, max_df=0.5)
x = bow.fit_transform(amazon['Text']).toarray()
features = np.array(bow.get_feature_names())
x = x[:, features != 'br']
features = features[features != 'br']
x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(x, amazon['Score'], test_size=0.2, random_state=0)

n_grid = [1000, 2500, 5000, 10000, 25000, 50000, 100000]
output = []
for n in n_grid:
	print('Current n:', n)
	x_train = x_train_full[:int(n*0.8)]
	y_train = y_train_full[:int(n*0.8)]
	x_test = x_test_full[:int(n*0.2)]
	y_test = y_test_full[:int(n*0.2)]
	if n <= 10000:
		output.append(cvfit(NN_layers=[1, 3, 5], NN_size=[100, 250, 500], es_epochs=500, LLE_var=[0.1, 1, 10, 100, 1000], n_estimators=[10, 50, 100]))
	else:
		output.append(cvfit(NN_layers=[1, 3, 5], NN_size=[100, 250, 500], es_epochs=100, LLE_var=[0.1, 1, 10, 100, 1000], n_estimators=[10, 50, 100]))

with open('results/amazon.pkl', 'wb') as f:
	pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

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

