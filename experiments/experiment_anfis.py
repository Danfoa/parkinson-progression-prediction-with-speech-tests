import time
import pandas as pd
from utils.dataset_loader import ParkinsonDataset
import numpy as np
from sklearn.metrics import accuracy_score
from clustering_models.som import SelfOrganizingMap
from clustering_models.em import ExpectationMaximization
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.decomposition import (
    PCA,
    IncrementalPCA
)


"""
Each regression model is going to be tested with different pre-processing and reduction techniques and hyper-parameters
the below script will contain all the model independent code.
"""

PCA_NUMBER_OF_FEATURES = 2
CROSS_VALIDATION_S = 1
SOM_CLUSTERS = 9  # According to Nilashi2019 paper
EM_CLUSTERS = 13  # According to Nilashi2019 paper


def anfis_model(self):
    experiment_name = "ANFIS Model"
    exp_start_time = time.process_time()

    # som_model, som_assignations = __train_som_model(self.training_sets, num_clusters=self.SOM_CLUSTERS)



    clusters = defaultdict(list)
    for instance, label, cluster in zip(self.training_sets, self.udprs, em_assignations):
        clusters[cluster].append(instance)

    results = defaultdict(list)
    i = 0
    for cluster in clusters.values():
        initial_features = len(cluster[0])
        print("Cluster no={}".format(i))

        n_components = self.PCA_NUMBER_OF_FEATURES
        for x in range(2, initial_features):
            pca = PCA(n_components=x)
            pca.fit(cluster)
            variances = pca.explained_variance_ratio_
            variances[x - 1] = variances[x - 1] + variances[x - 2]
            if variances[x] >= 0.9:
                n_components = x
                break

    print()


def __train_som_model(self, data, num_clusters=None):
    model = SelfOrganizingMap(data, num_clusters)
    assignations = model.clusterize()
    return model, assignations


def __train_em_model(data):
    model, assignations = ExpectationMaximization(data).fit_tranform()
    return model, assignations


if __name__ == '__main__':

    # Example of loading the dataset
    df = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                       return_gender=False)
    df.sample(frac=1)

    label_updrs = df["motor_UPDRS"]
    label_total_udprs = df[["total_UPDRS"]]

    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=StandardScaler(),
                                                             inplace=True)

    # # Split dataset
    X, X_train, X_test, y_train, y_test = ParkinsonDataset.split_dataset(dataset=df,
                                                                      subject_partitioning=False)

    # Step 1: EM - SOM clustering
    em_model, em_assignations = __train_em_model(X)


