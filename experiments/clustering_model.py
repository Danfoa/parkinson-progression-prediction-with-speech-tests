import os

import frange as frange
import numpy as np
from sklearn.decomposition import (
    PCA
)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from clustering_models.em import ExpectationMaximization
from clustering_models.som import SelfOrganizingMap
from regression_models import amfis_model
from utils.dataset_loader import ParkinsonDataset
from matplotlib import pyplot as plt
import decimal


"""
Each regression model is going to be tested with different pre-processing and reduction techniques and hyper-parameters
the below script will contain all the model independent code.
"""

PCA_NUMBER_OF_FEATURES = 2
CROSS_VALIDATION_S = 1
SOM_CLUSTERS = 9  # According to Nilashi2019 paper
EM_CLUSTERS = 13  # According to Nilashi2019 paper


def get_reduced_dataset(cluster_data, all_data):
    pca = PCA(.95)
    pca.fit_transform(cluster_data)
    return pca.transform(all_data)


def __train_som_model(data, num_clusters=None):
    model = SelfOrganizingMap(data, num_clusters)
    assignations = model.clusterize()
    return model, assignations


def __train_em_model(data):
    model, assignations = ExpectationMaximization(data).fit_tranform()
    return model, assignations


def get_reduced_dataset(cluster_data, all_data):
    pca = PCA(.9)
    pca.fit_transform(cluster_data)
    return pca.transform(all_data)

def find_accuracy(d, y, y_test, title):
    mse_f = np.mean(d ** 2)
    mae_f = np.mean(abs(d))
    rmse_f = np.sqrt(mse_f)
    r2_f = 1 - (sum(d ** 2) / sum((y - np.mean(y)) ** 2))

    print("Results for {}".format(title))
    print("MAE:", mae_f)
    print("MSE:", mse_f)
    print("RMSE:", rmse_f)
    print("R-Squared:", r2_f)
    print()

    x = list(range(len(y)))
    # plt.scatter(x, y, color="blue", label="original")
    # plt.plot(x, y_test, color="red", label="predicted")
    # plt.legend()
    # plt.title(title)
    # plt.show()
    return mae_f, r2_f


if __name__ == '__main__':
    # Example of loading the dataset
    df = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                       return_gender=False)
    df.sample(frac=1)

    label_updrs = df["motor_UPDRS"]
    label_total_udprs = df[["total_UPDRS"]]

    # Normalizing/scaling  dataset
    data, feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                                   scaler=MinMaxScaler(),
                                                                   inplace=False)

    # # Split dataset
    X, X_train, X_test, y_train, y_test = ParkinsonDataset.split_dataset(dataset=data,
                                                                         subject_partitioning=False)

    path = os.path.join("..", "reduced_datasets/predict_")
    np.save(path + "init_results", y_test)

    x_train = get_reduced_dataset(X_train, X_train)
    x_test = get_reduced_dataset(X_train, X_test)

    model = amfis_model.AMFIS(x_train, y_train)
    model.fit(k=0.0523, gamma=5000)
    path = os.path.join("..", "reduced_datasets/predict_")
    y = model.predict(x_test)
    np.save(path + "init", y)

    d0 = y_test[:, 0] - y[:, 0]
    d1 = y_test[:, 1] - y[:, 1]
    mae_t, r_t = find_accuracy(d0, y[:, 0], y_test[:, 0], "total_UPDRS")
    mae, r = find_accuracy(d1, y[:, 1], y_test[:, 1], "motor_UPDRS")
    print()
    print()






