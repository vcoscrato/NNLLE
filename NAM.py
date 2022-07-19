from __future__ import division

from typing import Union, Iterable, Sized, Tuple
from numpy.lib.function_base import append

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.optim import Adamax as optimm

import numpy as np
import time
import itertools
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import collections


class ExU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(in_dim))
        self.init_params()

    
    def init_params(self):
        self.weight = nn.init.normal_(self.weight, mean=4., std=.5)
        self.bias = nn.init.normal_(self.bias, std=.5)

    
    def forward(self, x):
        out = torch.matmul((x - self.bias), torch.exp(self.weight))
        out = torch.clamp(out, 0, 1)
        return out


class ReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_params()


    def init_params(self):        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.linear.bias, std=.5)


    def forward(self, x):
        out = self.linear(x)
        out = F.relu(out)
        return out



class FeatureNet(nn.Module):
    def __init__(self, dropout_rate, use_exu):
        super().__init__()
        if use_exu:
            layers = [ExU(1, 1024), nn.Dropout(dropout_rate), 
                      nn.Linear(1024, 1, bias = False)]
        else:
            layers = [ReLU(1, 64), nn.Dropout(dropout_rate),
                      ReLU(64, 64), nn.Dropout(dropout_rate),
                      ReLU(64, 32), nn.Dropout(dropout_rate),
                      nn.Linear(32, 1, bias = False)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NeuralAdditiveModel(nn.Module):
    def __init__(self, no_features, dropout_rate = .0, feature_dropout = .0, use_exu = False):
        super().__init__()
        self.no_features = no_features
        feature_nets = [FeatureNet(dropout_rate, use_exu) for _ in range(no_features)]
        self.feature_nets = nn.ModuleList(feature_nets)
        self.feature_drop = nn.Dropout(feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1,), requires_grad=True)
            
    def forward(self, x):
        y = []
        for i in range(self.no_features):
            o = self.feature_nets[i](x[:,i].unsqueeze(1))
            y.append(o)
        y = torch.cat(y, 1)
        y = self.feature_drop(y)
        out = torch.sum(y, axis = -1) + self.bias
        return out, y
    
    
    

    

def _np_to_tensor(arr):
    arr = np.array(arr, dtype='f4')
    arr = torch.from_numpy(arr)
    return arr


class NAM(BaseEstimator):

    def __init__(self,
                 shallow=False,
                 nn_weight_decay=0,
                 nn_dropout=0,

                 es = True,
                 es_validation_set_size = 0.1,
                 es_splitter_random_state = 0,

                 nepoch=200,
                 optim_lr=1e-3,

                 batch_size=1024,
                 batch_test_size=2000,
                 
                 dataloader_workers=1,
                 gpu=False,
                 verbose=5,

                 scale_data=True,
                 feature_penalty=0.0,
                 feature_dropout=0.0
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def fit(self, x_train, y_train):
        x_train = np.array(x_train, copy=False)
        y_train = np.array(y_train, copy=False)

        if self.scale_data:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)
        self.gpu = self.gpu and torch.cuda.is_available()

        self.x_dim = x_train.shape[1]

        self._construct_neural_net()
        self.epoch_count = 0

        if self.gpu:
            self.move_to_gpu()
            
        self.criterion = nn.MSELoss()

        return self.improve_fit(x_train, y_train, self.nepoch)

    def move_to_gpu(self):
        self.neural_net.cuda()
        self.gpu = True

        return self

    def move_to_cpu(self):
        self.neural_net.cpu()
        self.gpu = False

        return self

    def improve_fit(self, x_train, y_train, nepoch=1):
        x_train = np.array(x_train, copy=False)
        y_train = np.array(y_train, copy=False)

        if self.scale_data:
            x_train = self.scaler.transform(x_train)

        assert(self.batch_size >= 1)
        assert(self.batch_test_size >= 1)

        y_dtype = "f4"
        inputv_train = np.array(x_train, dtype="f4")
        target_train = np.array(y_train, dtype=y_dtype)

        range_epoch = range(nepoch)
        if self.es:
            splitter = ShuffleSplit(n_splits=1,
                test_size=self.es_validation_set_size,
                random_state=self.es_splitter_random_state)
            index_train, index_val = next(iter(splitter.split(x_train,
                y_train)))
            self.index_train = index_train
            self.index_val = index_val

            inputv_val = inputv_train[index_val]
            target_val = target_train[index_val]
            inputv_val = np.ascontiguousarray(inputv_val)
            target_val = np.ascontiguousarray(target_val)

            inputv_train = inputv_train[index_train]
            target_train = target_train[index_train]
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            self.best_loss_val = np.infty
            es_tries = 0
            range_epoch = itertools.count() # infty iterator

            batch_test_size = min(self.batch_test_size, inputv_val.shape[0])
            self.loss_history_validation = []

        self.loss_history_train = []
        start_time = time()

        self.actual_optim_lr = self.optim_lr
        optimizer = optimm(
            self.neural_net.parameters(),
            lr=self.actual_optim_lr,
            weight_decay=self.nn_weight_decay
        )
        err_count = 0
        es_penal_tries = 0
        
        for _ in range_epoch:
            self.cur_batch_size = min(self.batch_size, inputv_train.shape[0])

            permutation = np.random.permutation(target_train.shape[0])
            inputv_train = torch.from_numpy(inputv_train[permutation])
            target_train = torch.from_numpy(target_train[permutation])
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            try:
                self.neural_net.train()
                self._one_epoch(True, self.cur_batch_size, inputv_train,
                                target_train, optimizer)
                optimizer.param_groups[0]['lr'] *= 0.995
                self.epoch_count += 1

                if self.es:
                    self.neural_net.eval()
                    avloss = self._one_epoch(False, batch_test_size,
                        inputv_val, target_val, optimizer)
                    self.loss_history_validation.append(avloss)
                    if avloss <= self.best_loss_val:
                        self.best_loss_val = avloss
                        best_state_dict = self.neural_net.state_dict()
                        best_state_dict = deepcopy(best_state_dict)
                        es_tries = 0
                        if self.verbose >= 2:
                            print("This is the lowest validation loss so far.")
                        self.best_loss_history_validation = avloss
                    else:
                        es_tries += 1
                        if self.verbose >= 2:
                            print(f'No improvement for {es_tries} epochs. Restarting from best route.')
                        self.neural_net.load_state_dict(best_state_dict)

                    if es_tries > 10:
                        if self.verbose >= 2:
                            print("Validation loss did not improve after 10 tries. Stopping.")
                            break
                        
            except KeyboardInterrupt:
                if self.epoch_count > 0 and self.es:
                    print("Keyboard interrupt detected.",
                          "Switching weights to lowest validation loss",
                          "and exiting")
                    self.neural_net.load_state_dict(best_state_dict)
                break

        self.elapsed_time = time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", self.elapsed_time, flush=True)

        return self

    def _one_epoch(self, is_train, batch_size, inputv, target,
        optimizer):
        with torch.set_grad_enabled(is_train):
            inputv = torch.from_numpy(inputv)
            target = torch.from_numpy(target)

            loss_vals = []
            batch_sizes = []
            
            tdataset = data.TensorDataset(inputv, target)
            data_loader = data.DataLoader(tdataset,
                batch_size=batch_size, shuffle=True, drop_last=is_train,
                pin_memory=self.gpu, num_workers=self.dataloader_workers)

            for inputv_this, target_this in data_loader:
                if self.gpu:
                    inputv_this = inputv_this.cuda(non_blocking=True)
                    target_this = target_this.cuda(non_blocking=True)
                
                inputv_this.requires_grad_(True)
                batch_actual_size = inputv_this.shape[0]

                optimizer.zero_grad()
                output, thetas = self.neural_net(inputv_this)

                loss = self.criterion(output, target_this)
                np_loss = loss.data.item()
                if np.isnan(np_loss):
                    raise RuntimeError("Loss is NaN")

                loss_vals.append(np_loss)
                batch_sizes.append(batch_actual_size)

                if is_train:
                    loss += self.feature_penalty * (thetas ** 2).mean() # Add feature penalty
                    loss.backward()
                    optimizer.step()

            avgloss = np.average(loss_vals, weights=batch_sizes)
            if self.verbose >= 2:
                print("Finished epoch", self.epoch_count,
                      "with batch size", batch_size, "and",
                      ("train" if is_train else "validation"),
                      "loss", avgloss, flush=True)

            return avgloss

    def score(self, x_test, y_test):

        if self.scale_data:
            x_test = self.scaler.transform(x_test)

        with torch.no_grad():
            self.neural_net.eval()
            inputv = _np_to_tensor(np.ascontiguousarray(x_test))
            target = _np_to_tensor(np.ascontiguousarray(y_test))

            batch_size = min(self.batch_test_size, x_test.shape[0])

            loss_vals = []
            batch_sizes = []

            tdataset = data.TensorDataset(inputv, target)
            data_loader = data.DataLoader(tdataset,
                batch_size=batch_size, shuffle=True, drop_last=False,
                pin_memory=self.gpu, num_workers=self.dataloader_workers,)

            for inputv_this, target_this in data_loader:
                if self.gpu:
                    inputv_this = inputv_this.cuda(non_blocking=True)
                    target_this = target_this.cuda(non_blocking=True)

                batch_actual_size = inputv_this.shape[0]
                output, _ = self.neural_net(inputv_this)
                
                loss = self.criterion(output, target_this)

                loss_vals.append(loss.data.item())
                batch_sizes.append(batch_actual_size)

            return -1 * np.average(loss_vals, weights=batch_sizes)

    def predict(self, x_pred, return_thetas=False):
        
        if self.scale_data:
            x_pred = self.scaler.transform(x_pred)

        with torch.no_grad():
            self.neural_net.eval()
            inputv = _np_to_tensor(x_pred)

            if self.gpu:
                inputv = inputv.cuda()

            output_pred, thetas = self.neural_net(inputv)
            
            if return_thetas:
                return output_pred.data.cpu().numpy(), thetas.data.cpu().numpy()
            else:
                return output_pred.data.cpu().numpy()

    def _construct_neural_net(self):
        
        self.neural_net = NeuralAdditiveModel(no_features=self.x_dim,
                                              dropout_rate=self.nn_dropout,
                                              use_exu=self.shallow, 
                                              feature_dropout=self.feature_dropout)

        
        
if __name__ == "__main__":
    
    import pandas as pd
    import numpy as np
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
    
    # Sanity check
    from sklearn.datasets import fetch_california_housing
    dataset = fetch_california_housing()
    x = dataset.data
    y = dataset.target
    
    pred = cross_val_predict(NAM(shallow=False, optim_lr=0.00674, feature_penalty=0.001, 
                                  nn_weight_decay=10**-6, verbose=5), 
                              x, y, cv=KFold(n_splits=5, shuffle=True))
    print(pred[:5], y[:5])
    print(f'Mean squared error: {np.sqrt(mean_squared_error(y, pred))}. Reported in the paper: 0.562')
    
    
    # Housing
    print('Housing...')
    dataset = pd.read_csv('data/housing.csv', sep='\s+', header=None)
    dataset = dataset.sample(frac=1, random_state=0)
    x = dataset.iloc[:, range(0, dataset.shape[1] - 1)].values
    y = dataset.iloc[:, -1].values

    parameters = [{'shallow': [True, False], 'feature_dropout': [0, 0.1, 0.2], 'feature_penalty': [0, 0.1, 0.2]}]
    t0 = time()
    pred = cross_val_predict(GridSearchCV(NAM(verbose=0), parameter, scoring='neg_mean_squared_error', iid=False, 
                             cv=KFold(n_splits=2, shuffle=False, random_state=0)), x, y, cv=KFold(n_splits=5, shuffle=False))
    t = time() - t0
    
    mse = mean_squared_error(y, pred)
    mse_std = np.std((pred.flatten()-y)**2) / (len(y)**(1/2))
    mae = mean_absolute_error(y, pred)
    mae_std = np.std(abs(pred.flatten()-y)) / (len(y)**(1/2))

    with open('fitted/housing_NAM4.pkl', 'wb') as f:
        pickle.dump(pred, f, pickle.HIGHEST_PROTOCOL)
    with open('results/housing_NAM4.txt', 'w') as f:
        print([mse, mse_std, mae, mae_std, t], file=f)

        
    # Superconductivity
    print('Superconductivity...')
    dataset = pd.read_csv('data/superconductivity.csv')
    dataset = dataset.sample(frac=1, random_state=0)
    x = dataset.iloc[:, range(0, dataset.shape[1] - 1)].values
    y = dataset.iloc[:, -1].values
    
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=0)
    
    output = []
    best_mse = np.infty
    t0 = time()
    print('NLS\'s...')
    for shallow in [False]:
        print('Current model is shallow:', shallow)
        for dropout in [0]:
            print('Current dropout:', dropout)
            for penalty in [0.2]:
                print('Current penalty:', penalty)
                model = NAM(shallow=shallow, verbose=2, feature_dropout=dropout, feature_penalty=penalty).fit(x_train, y_train)
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
    
    with open('fitted/superconductivity_NAM2.pkl', 'wb') as f:
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
    with open('results/superconductivity_NAM2.txt', 'w') as f:
        print(output, file=f)

        
    # BlogFeedback
    print('Blog...')
    dataset = pd.read_csv('data/blog feedback.csv')
    x = dataset.iloc[:, range(0, dataset.shape[1] - 1)].values
    y = dataset.iloc[:, -1].values
    
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=0)
    
    output = []
    best_mse = np.infty
    t0 = time()
    print('NLS\'s...')
    for shallow in [True, False]:
        print('Current model is shallow:', shallow)
        for dropout in [0, 0.1, 0.2]:
            print('Current dropout:', dropout)
            for penalty in [0, 0.1, 0.2]:
                print('Current penalty:', penalty)
                model = NAM(shallow=shallow, verbose=3, optim_lr=1e-2, feature_dropout=dropout, feature_penalty=penalty).fit(x_train, y_train)
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

    with open('fitted/blogfeedback_NAM.pkl', 'wb') as f:
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
    with open('results/blogfeedback_NAM.txt', 'w') as f:
        print(output, file=f)
