import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from neupy import algorithms
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal



class ExpectationMaximization:

    def __init__(self, data):
        self.data = data
        self.model = None
        self.transformed_input = None

    def fit_tranform(self):
        models = []
        n_components = np.arange(1, 21)
        models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(self.data) for n in n_components]

        aic = [m.aic(self.data) for m in models]
        bic = [m.bic(self.data) for m in models]

        self.model = models[np.argmin(bic)]

        plt.plot(n_components, aic, label='AIC')
        plt.plot(n_components, bic, label='BIC')

        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.savefig("out/EM_BIC_AIC.png")
        plt.close()

        self.model.fit(self.data)
        print('Converged:',  self.model.converged_)  # Check if the model has converged
        labels = self.model.predict(self.data)


        return self.model, labels



