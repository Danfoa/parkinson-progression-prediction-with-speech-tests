import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from neupy import algorithms


class SelfOrganizingMap:

    def __init__(self, dataset, nclusters):
        self.dataset = dataset
        self.number_of_features = dataset.shape[1]
        self.number_of_rows = dataset.shape[0]
        self.number_of_clusters = nclusters
        self.model = None
        self.__fit()
        self.__transform()

    def __fit(self):
        self.model = algorithms.SOFM(
            n_inputs=self.number_of_features, n_outputs=self.number_of_clusters, weight='sample_from_data')

    def __transform(self):
        self.model.train(self.dataset.to_numpy(), epochs=200)

    def model_name(self):
        return "SOM"

    def clusterize(self):
        som_results = self.model.predict(self.dataset)
        # Decode one-hot encoded results
        normalized_results = [np.where(r==1)[0][0] for r in som_results]
        return normalized_results