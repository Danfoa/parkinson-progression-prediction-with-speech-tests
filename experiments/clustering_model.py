import time
import pandas as pd
import os
from utils.dataset_loader import ParkinsonDataset
import numpy as np
from sklearn.metrics import accuracy_score
from clustering_models.som import SelfOrganizingMap
from clustering_models.em import ExpectationMaximization
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from matplotlib import pyplot as plt
from collections import defaultdict
from regression_models import amfis_model
from sklearn.model_selection import train_test_split

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


def __train_som_model(data, num_clusters=None):
    model = SelfOrganizingMap(data, num_clusters)
    assignations = model.clusterize()
    return model, assignations

def __train_em_model(data):
    model, assignations = ExpectationMaximization(data).fit_tranform()
    return model, assignations

def get_reduced_dataset(cluster_data, all_data):
    pca = PCA(.95)
    pca.fit_transform(cluster_data)
    return pca.transform(all_data)


if __name__ == '__main__':
    # Example of loading the dataset
    df = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                       return_gender=False)
    df.sample(frac=1)

    label_updrs = df["motor_UPDRS"]
    label_total_udprs = df[["total_UPDRS"]]

    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=MinMaxScaler(),
                                                             inplace=True)

    # # Split dataset
    X, X_train, X_test, y_train, y_test = ParkinsonDataset.split_dataset(dataset=df,
                                                                      subject_partitioning=False)

    # Step 1: em - SOM clustering
    som_model, som_assignations = __train_som_model(X, SOM_CLUSTERS)

    em_model, em_assignations = __train_em_model(X)

    em_number_of_clusters = em_model.n_components

    clusters = defaultdict(list)
    for instance, cluster in zip(X, em_assignations):
        clusters[cluster].append(instance)


    pca_models = []
    load_path = "../../results/clustering/"
    save_path = "../../results/reduction/"
    results = defaultdict(list)
    clustering_algorithms = ['em', 'som']
    for cluster in clusters.values():
        initial_features = len(cluster[0])

        n_components = PCA_NUMBER_OF_FEATURES
        for x in range(n_components, initial_features):
            pca = PCA(n_components=x)
            pca.fit(cluster)
            variances = pca.explained_variance_ratio_
            variances = np.cumsum(variances)
            res = list(filter(lambda i: i > 0.9, list(variances)))
            if res:
                n_components = x
                pca_models.append(pca)
                break


    X_training_sets = []
    path = os.path.join("..", "reduced_datasets/datasets_")
    np.save(path + 'y', y_train)
    for i in range(em_number_of_clusters):
        pca_model = pca_models[i]
        x_train_cluster = pca_model.transform(X_train)
        X_training_sets.append(x_train_cluster)
        np.save(path + str(i), x_train_cluster)

        print()

    i = 0
    # X_train, X_test, y_train, y_test
    for x_train_cluster in X_training_sets:
        # # Split dataset

        model = amfis_model.AMFIS(x_train_cluster, y_train)
        model.fit()
        path = os.path.join("..", "reduced_datasets/predict_")
        y = model.predict(X_test)
        np.save(path + str(i), y)
        i += 1

