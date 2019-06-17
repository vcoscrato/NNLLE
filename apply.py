import numpy as np
from nnlocallinear import NNPredict, LLE
from sklearn.metrics import mean_squared_error as mse


def apply(x_train, y_train, x_test, y_test, hidden_size, num_layers, varying_theta0, fixed_theta0, penalty, fit_LLE):

    lowest_loss = np.infty

    for layers in num_layers:

        for hidden in hidden_size:

            print('Fitting NN: layers = ', layers, ', hidden = ', hidden)
            model = NNPredict(
                verbose=0,
                es=True,
                es_give_up_after_nepochs=50,
                hidden_size=hidden,
                num_layers=layers,
                gpu=False,
                tuningp=penalty,
                scale_data=True,
                varying_theta0=varying_theta0,
                fixed_theta0=fixed_theta0,
                dataloader_workers=0).fit(x_train, y_train)

            loss = mse(y_test, model.predict(x_test))

            if loss < lowest_loss:

                lowest_loss = loss
                best_model = model

    print('Fitting LLE')
    if fit_LLE:

        model2 = LLE().fit(x_train, y_train)
        loss = mse(y_test, model2.predict(x_test, 1))

    print('lowest_loss = ', lowest_loss)

    if fit_LLE:

        print('LLE_loss = ', loss)
        return [best_model, model2]

    return model
