import time
from utils.dataset_loader import ParkinsonDataset
import numpy as np
from sklearn.metrics import accuracy_score
from som import SelfOrganizingMap
from em import ExpectationMaximization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        scaler = StandardScaler()
        msk = np.random.rand(len(self.dataset)) < 0.6
        self.training_sets = self.dataset[msk]
        self.udprs = self.training_sets["motor_UPDRS"]
        self.total_udprs = self.training_sets["total_UPDRS"]
        self.training_sets = self.training_sets.drop(['subject#', 'motor_UPDRS','total_UPDRS'], axis=1)
        scaler.fit(self.training_sets)
        self.training_sets = scaler.fit_transform(self.training_sets)

        scaler = StandardScaler()
        self.test_sets = self.dataset[~msk]
        self.y_udprs = self.test_sets["motor_UPDRS"]
        self.y_total_udprs = self.test_sets["total_UPDRS"]
        self.test_sets = self.test_sets.drop(['subject#', 'motor_UPDRS','total_UPDRS'], axis=1)
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
            clusters[cluster].append(instance)

        results = defaultdict(list)
        i = 0
        for cluster in clusters.values():
            initial_features = len(cluster[0])
            print("Cluster no={}".format(i))

            n_components = self.PCA_NUMBER_OF_FEATURES
            for x in range(2, initial_features):
                pca = PCA(n_components = x)
                pca.fit(cluster)
                variances = pca.explained_variance_ratio_
                variances[x-1] = variances[x-1] + variances[x-2]
                if variances[x] >= 0.9:
                    n_components = x
                    break

            cluster = pca.transform(cluster)


            # amfis = AMFIS(cluster)
            # result = amfis.learn()


        # Plotting the Cumulative Summation of the Explained Variance

        print()


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
