import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import os



class ExpectationMaximization:

    def __init__(self, data):
        self.data = data
        self.model = None
        self.transformed_input = None

    def fit_tranform(self, best_n_components=12):
        n_components = np.arange(1, 21)
        models = [GaussianMixture(n, covariance_type='full', max_iter=400)
                      .fit(self.data) for n in n_components]

        aic = [m.aic(self.data) for m in models]
        bic = [m.bic(self.data) for m in models]

        min_idx_bic = int(np.argmin(bic))
        min_idx_aic = int(np.argmin(aic))
        self.model = models[best_n_components]

        plt.plot(n_components, aic, label='AIC')
        plt.plot(n_components, bic, label='BIC')

        plt.legend(loc='best')
        plt.xlabel('n_components')
        path = os.path.join('..', 'out/EM_BIC_AIC.png')
        plt.savefig(path)
        plt.close()

        self.model.fit(self.data)
        probabilities = self.model.predict_proba(self.data)
        path = os.path.join('..', 'results/clustering/EM/')
        np.save(path + 'probabilities', probabilities)
        np.save(path + 'aic', aic)
        np.save(path + 'bic', bic)

        print('Converged:',  self.model.converged_)  # Check if the model has converged
        labels = self.model.predict(self.data)
        np.save(path + 'assignmetnts', labels)


        return self.model, labels
