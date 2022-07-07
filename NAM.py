from __future__ import division

from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F

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


def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("abstract method called")


class ExULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return torch.clip((x - self.bias) @ torch.exp(self.weight), 0, 1)


class ReLULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)


class FeatureNN(torch.nn.Module):
    def __init__(self, shallow:bool):
        super().__init__()
        if shallow:
            self.layers = [ExULayer(1, 1024)]
            self.linear = torch.nn.Linear(1024, 1, bias=False)
        else:
            self.layers = [ReLULayer(1, 64), ReLULayer(64, 64), ReLULayer(64, 32)]
            self.linear = torch.nn.Linear(32, 1, bias=False)

        self.dropout = torch.nn.Dropout(p=0)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.linear(x)


class NeuralAdditiveModel(torch.nn.Module):
    def __init__(self, input_size: int, shallow: bool):
        super().__init__()
        self.input_size = input_size

        self.feature_nns = torch.nn.ModuleList([
            FeatureNN(shallow)
            for i in range(input_size)
        ])
        self.feature_dropout = torch.nn.Dropout(p=0)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f_out = torch.cat(self._feature_nns(x), dim=-1)
        f_out = self.feature_dropout(f_out)

        return f_out.sum(axis=-1) + self.bias, f_out

    def _feature_nns(self, x):
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]
    

def _np_to_tensor(arr):
    arr = np.array(arr, dtype='f4')
    arr = torch.from_numpy(arr)
    return arr


class NAM(BaseEstimator):

    def __init__(self,
                 shallow=False,
                 nn_weight_decay=0,

                 es = True,
                 es_validation_set_size = None,
                 es_give_up_after_nepochs = 10,
                 es_splitter_random_state = 0,

                 nepoch=200,

                 batch_initial=300,
                 batch_step_multiplier=1.4,
                 batch_step_epoch_expon=2.0,
                 batch_max_size=1000,

                 optim_lr=1e-2,

                 dataloader_workers=1,
                 batch_test_size=2000,
                 gpu=False,
                 verbose=5,

                 scale_data=True,
                 feature_penalty=0.0
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

        assert(self.batch_initial >= 1)
        assert(self.batch_step_multiplier > 0)
        assert(self.batch_step_epoch_expon > 0)
        assert(self.batch_max_size >= 1)
        assert(self.batch_test_size >= 1)

        y_dtype = "f4"
        inputv_train = np.array(x_train, dtype="f4")
        target_train = np.array(y_train, dtype=y_dtype)

        range_epoch = range(nepoch)
        if self.es:
            es_validation_set_size = self.es_validation_set_size
            if es_validation_set_size is None:
                es_validation_set_size = round(
                    min(x_train.shape[0] * 0.10, 5000))
            splitter = ShuffleSplit(n_splits=1,
                test_size=es_validation_set_size,
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

        batch_max_size = min(self.batch_max_size, inputv_train.shape[0])
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
            self.cur_batch_size = int(min(batch_max_size,
                self.batch_initial +
                self.batch_step_multiplier *
                self.epoch_count ** self.batch_step_epoch_expon))

            permutation = np.random.permutation(target_train.shape[0])
            inputv_train = torch.from_numpy(inputv_train[permutation])
            target_train = torch.from_numpy(target_train[permutation])
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            try:
                self.neural_net.train()
                self._one_epoch(True, self.cur_batch_size, inputv_train,
                                target_train, optimizer)

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
                            print("This is the lowest validation loss",
                                  "so far.")
                        self.best_loss_history_validation = avloss
                    else:
                        es_tries += 1

                    if (es_tries == self.es_give_up_after_nepochs
                        // 3 or
                        es_tries == self.es_give_up_after_nepochs
                        // 3 * 2):
                        if self.verbose >= 2:
                            print("No improvement for", es_tries,
                             "tries")
                            print("Decreasing learning rate by half")
                            print("Restarting from best route.")
                        optimizer.param_groups[0]['lr'] *= 0.5
                        self.neural_net.load_state_dict(
                            best_state_dict)
                    elif es_tries >= self.es_give_up_after_nepochs:
                        self.neural_net.load_state_dict(
                            best_state_dict)
                        if self.verbose >= 1:
                            print(
                                "Validation loss did not improve after",
                                self.es_give_up_after_nepochs,
                                "tries. Stopping"
                            )
                        break

                self.epoch_count += 1
            except RuntimeError as err:
                #if self.epoch_count == 0:
                #    raise err
                if self.verbose >= 2:
                    print("Runtime error problem probably due to",
                           "high learning rate.")
                    print("Decreasing learning rate by half.")

                self._construct_neural_net()
                if self.gpu:
                    self.move_to_gpu()
                self.actual_optim_lr /= 2
                optimizer = optimm(
                    self.neural_net.parameters(),
                    lr=self.actual_optim_lr,
                    weight_decay=self.nn_weight_decay
                )
                self.epoch_count = 0

                continue
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
                    loss += self.feature_penalty * (thetas ** 2).sum() # Add feature penalty
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
                
                print(output, target_this)
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
        
        self.neural_net = NeuralAdditiveModel(input_size=self.x_dim, shallow=self.shallow)

        
        
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
    
    
    # Housing
    print('Housing...')
    dataset = pd.read_csv('data/housing.csv', sep='\s+', header=None)
    dataset = dataset.sample(frac=1, random_state=0)
    x = dataset.iloc[:, range(0, dataset.shape[1] - 1)].values
    y = dataset.iloc[:, -1].values

    '''
    # Superconductivity
    print('Superconductivity...')
    dataset = pd.read_csv('data/superconductivity.csv')
    dataset = dataset.sample(frac=1, random_state=0)
    x = dataset.iloc[:, range(0, dataset.shape[1] - 1)].values
    y = dataset.iloc[:, -1].values
    '''
    
    model = NAM(shallow=False, verbose=5)
    model.fit(x, y)
    print(model.predict(x[:5], return_thetas=False))
    print(np.mean(np.abs(model.predict(x) - y)))
