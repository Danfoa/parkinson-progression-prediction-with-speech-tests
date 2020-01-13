import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from sklearn.mixture import GaussianMixture
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


class ExpectationMaximization:

    def __init__(self, data):
        self.data = data
        self.model = None
        self.transformed_input = None

    def fit_tranform(self, best_n_components=None):
        n_components = np.arange(1, 21)
        models = [GaussianMixture(n, covariance_type='full', max_iter=400)
                      .fit(self.data) for n in n_components]

        aic = [m.aic(self.data) for m in models]
        bic = [m.bic(self.data) for m in models]

        min_idx_bic = int(np.argmin(bic))
        min_idx_aic = int(np.argmin(aic))
        self.model = models[best_n_components]

        # plt.plot(n_components, aic, label='AIC')
        # plt.plot(n_components, bic, label='BIC')


        metric = pd.DataFrame(data={
                                'BIC': bic,
                                'AIC': aic})

        x = np.arange(1, 21,3)
        sns.lineplot(data=metric, color="coral")
        plt.xticks(x)
        plt.tight_layout()

        plt.show()

        path = os.path.join('..', 'media/clustering/em/EM_BIC_AIC.png')
        plt.savefig(path)
        plt.close()

        self.model.fit(self.data)
        probabilities = self.model.predict_proba(self.data)
        path = os.path.join('..', 'results/clustering/em/')
        np.save(path + 'probabilities', probabilities)
        np.save(path + 'aic', aic)
        np.save(path + 'bic', bic)

        print('Converged:',  self.model.converged_)  # Check if the model has converged
        labels = self.model.predict(self.data)
        path = path + 'C={}-labels'.format(best_n_components)
        np.save(path, labels)


        return self.model, labels
