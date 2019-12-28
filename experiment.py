import time
import pandas as pd
from utils import DatasetLoader
import numpy as np
from sklearn.metrics import accuracy_score
from som import SelfOrganizingMap
from em import ExpectationMaximization
from sklearn.preprocessing import MinMaxScaler
from amfis_model import AMFIS
from regresion_neural_network import RegressionNeuralNetwork
from support_vector_machine_regression import SupportVectorMachineRegression
from cart_model import ClassificationRegressionTree
from matplotlib import pyplot as plt
from collections import defaultdict



from sklearn.decomposition import (
    PCA,
    IncrementalPCA
)

class Experiment:
    PCA_NUMBER_OF_FEATURES = 2
    CROSS_VALIDATION_S = 1
    SOM_CLUSTERS = 9  # According to Nilashi2019 paper
    EM_CLUSTERS = 13 # According to Nilashi2019 paper

    def __init__(self):
        self.dataset = None

        self.training_sets = []
        self.udprs = None
        self.total_udprs = None

        self.test_sets = []
        self.y_udprs = None
        self.y_total_udprs = None

        self.experiment_name = None

    def __split_dataset(self):
        scaler = MinMaxScaler()
        msk = np.random.rand(len(self.dataset)) < 0.8
        self.training_sets = self.dataset[msk]
        self.udprs = self.training_sets["motor_UPDRS"]
        self.total_udprs = self.training_sets["total_UPDRS"]
        self.training_sets = self.training_sets.drop(['subject#', 'motor_UPDRS','total_UPDRS', 'test_time'], axis=1)
        self.training_sets = scaler.fit_transform(self.training_sets)

        scaler = MinMaxScaler()
        self.test_sets = self.dataset[~msk]
        self.y_udprs = self.test_sets["motor_UPDRS"]
        self.y_total_udprs = self.test_sets["total_UPDRS"]
        self.test_sets = self.test_sets.drop(['subject#', 'motor_UPDRS','total_UPDRS', 'test_time'], axis=1)
        self.test_sets = scaler.fit_transform(self.test_sets)


    def announce_configuration(self, conf):
        ds = self.experiment_name
        print()
        print(" * Experiment {}".format(ds))

    def anfis_model(self):
        self.experiment_name = "ANFIS Model"
        self.__experiment_init()
        exp_start_time = time.process_time()


        # som_model, som_assignations = self.__train_som_model(self.training_sets, num_clusters=self.SOM_CLUSTERS)


        em_model, em_assignations = self.__train_em_model(self.training_sets)

        clusters = defaultdict(list)
        for instance, label, cluster in zip(self.training_sets, self.udprs, em_assignations):
            clusters[cluster].append((instance, label))

        results = defaultdict(list)
        for cluster in clusters.values():
            features = [i[0] for i in cluster]
            pca = PCA().fit(features)
            plt.figure()
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Variance (%)')  # for each component
            plt.title('Pulsar Dataset Explained Variance')
            plt.show()

            # amfis = AMFIS(cluster)
            # result = amfis.learn()


        # Plotting the Cumulative Summation of the Explained Variance


        pca_sklrn = PCA(self.training_sets, Experiment.PCA_NUMBER_OF_FEATURES)
        pca_sklrn_result = pca_sklrn.fit_transform(self.training_sets)



    def __experiment_init(self):
        self.dataset = DatasetLoader.load_dataset()

        self.__split_dataset()
        exp = self.experiment_name
        print("Experimentation: {}".format(exp))


    def __train_som_model(self, data, num_clusters = None):
        model = SelfOrganizingMap(data, num_clusters)
        assignations = model.clusterize()
        return model, assignations

    def __train_em_model(self, data):
        model, assignations = ExpectationMaximization(data).fit_tranform()
        return model, assignations
