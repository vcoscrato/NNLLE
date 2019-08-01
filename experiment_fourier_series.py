import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

from nnlocallinear.nnlocallinear import LLS, NLS


class FourierSeries(object):
    """
    Generate a sin series where
    f(x) = coefficients[0]*sin(frequencies[0]*x) + ...

    You can define the domain of the function using
    x_domain(delta_x, domain)

    """
    def __init__(self, coefficients=[1, 0.1], frequencies=[1, 10]):
        self.frequencies = np.array(frequencies)
        self.coefficients = coefficients

    def plot(self, x=None, model=None):
        if x is None:
            x = self.x_domain(0.001)
        # plt.scatter(x, self.f(x))
        plt.plot(x, self.f(x), label='Original function')
        if model is not None:
            plt.plot(x, model.predict(x))
        plt.xlabel('x value')
        plt.ylabel('f(x)')
        plt.show()

    def x_domain(self, delta_x, domain=[0, 2*np.pi]):
        """
        x_domain generate a numpy array shape(-1, 1) for be used as
        a input in the function.
        :param delta_x: x step
        :param domain: list domain to generate the x array - [max, min]
        :return: np array shape(-1, 1)
        """
        return np.arange(domain[0], domain[1], delta_x).reshape(-1, 1)

    def f(self, x):
        return np.sum(np.array([a_i*np.sin(f_i*x) for a_i, f_i in zip(self.coefficients, self.frequencies)]), axis=0)


if __name__ == '__main__':
    fs = FourierSeries()
    # fs.plot(x=fs.x_domain(0.001))
#     x_train = fs.x_domain(0.001)
#     Aqui esta o problema o fit depende muito 
#     do numero de pontos, quando uso um grid fino, que me da mais pontos ele fita razoavel
#     Mas mesmo assim nao consegui fitar a serie com 3 funcoes seno 
#   fs = FourierSeries(coefficients=[1, 0.1, 0.01], frequencies=[1, 10, 100])
    x_train = fs.x_domain(0.00001)

    y_train = fs.f(x_train)
    indices = np.random.permutation(len(y_train))

    print('Points:', len(y_train))

    x_train = x_train[indices]
    y_train = y_train[indices]

    model = NLS(
        verbose=2,
        es=True,
        es_give_up_after_nepochs=300,
        hidden_size=250,
        num_layers=4,
        gpu=True,
        scale_data=True,
        varying_theta0=False,
        fixed_theta0=True,
        penalization_thetas=0,
        dataloader_workers=0).fit(x_train, y_train)

    y_pred = model.predict(x_train)
    fs.plot(model=model)

    mse = metrics.mean_squared_error(y_train, y_pred)
    print(mse)

