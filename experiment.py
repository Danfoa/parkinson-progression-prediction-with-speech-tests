import time
import pandas as pd
from utils import DatasetLoader
import numpy as np
from sklearn.metrics import accuracy_score
from som import SelfOrganizingMap
from em import ExpectationMaximization
from amfis_model import AMFIS
from regresion_neural_network import RegressionNeuralNetwork
from support_vector_machine_regression import SupportVectorMachineRegression
from cart_model import ClassificationRegressionTree

from sklearn.decomposition import (
    PCA,
    IncrementalPCA
)

class Experiment:
    PCA_NUMBER_OF_FEATURES = 2

    def __init__(self):
        # dataset = df.drop(['subject#', 'motor_UPDRS','total_UPDRS', 'test_time'], axis=1)
        # udprs = df["motor_UPDRS"]
        # total_udprs = df["total_UPDRS"]

        self.dataset = None
        self.training_sets = []
        self.test_sets = []

        self.experiment_name = None

    def __split_dataset(self):
        msk = np.random.rand(len(self.dataset)) < 0.8
        self.training_sets = self.dataset[msk]
        self.test_sets = self.dataset[~msk]


    def __evaluate(self, configuration):
        accuracy_scores = []
        execution_times = []
        self.announce_configuration(configuration)
        conf_start_time = time.process_time()
        for trial in range(self.CROSS_VALIDATION_S):
            print("   - Processing trial #{:02d}".format(trial))
            training_set = self.training_sets[trial]
            test_set = self.test_sets[trial]
            acc, t = self.__evaluate_execution(configuration, training_set, test_set)
            print("     . Accuracy\t\t\t\t{}".format(acc))
            print("     . Execution time\t\t{} seconds".format(t))

            accuracy_scores.append(acc)
            execution_times.append(t)
        avg_accuracy = np.mean(accuracy_scores)
        avg_exec_time = np.mean(execution_times)
        print("   Average Accuracy {}".format(avg_accuracy))
        print("   Average Execution Time {}".format(avg_exec_time))

        conf_elapsed_time = time.process_time() - conf_start_time
        print("   Total Execution Time {}".format(conf_elapsed_time))
        return avg_accuracy, avg_exec_time


    def __evaluate_execution(self, configuration, training_set, test_set):
        trial_run_start_time = time.process_time()
        configuration.learn(training_set)
        trial_run_elapsed_time = time.process_time() - trial_run_start_time
        print("     . Learning time\t\t{}".format(trial_run_elapsed_time))
        trial_run_start_time = time.process_time()
        results = configuration.classify(test_set)
        y_labels = test_set[:, -1]
        trial_run_accuracy = accuracy_score(results, y_labels)
        trial_run_elapsed_time = time.process_time() - trial_run_start_time
        return trial_run_accuracy, trial_run_elapsed_time

    def announce_configuration(self, conf):
        ds = self.experiment_name
        print()
        print(" * Experiment {}".format(ds))

    def non_linear_model(self):
        self.experiment_name = "Non-Linear Model"
        self.__experiment_init()
        exp_start_time = time.process_time()

        # TODO
        model, assignations = self.__train_som_model(self.dataset)
        model, assignations = self.__train_em_model(self.dataset)

        # TODO:  LDA and PCA
        pca_sklrn = PCA(self.dataset, Experiment.PCA_NUMBER_OF_FEATURES)
        pca_sklrn_result = pca_sklrn.fit_transform(self.dataset.to_numpy())


        # Learn and Classify non Linear Models
        configurations = self.__build_configurations()
        for configuration in configurations:
            acc, exect = self.__evaluate(configuration)

    def __build_configurations(self):
        non_linear_algorithms = [
            AMFIS(),
            RegressionNeuralNetwork(),
            SupportVectorMachineRegression(),
            ClassificationRegressionTree()
        ]

        configurations =[]
        for d in non_linear_algorithms:
            configurations.append(d)
        return configurations


    def __num_of_rows(self):
        n_train = len(self.training_sets[0])
        n_test = len(self.test_sets[0])
        return n_train + n_test

    def __experiment_init(self):
        self.dataset = DatasetLoader.load_temporal_dataset()
        self.__split_dataset()

        num_rows = self.__num_of_rows()
        exp = self.experiment_name
        print("Experimentation: {}, in \033[0m (N={})".format(exp, self.experiment_name, num_rows))


    def __train_som_model(self, data, num_clusters):
        model = SelfOrganizingMap(data, num_clusters)
        assignations = model.clusterize()
        return model, assignations

    def __train_em_model(self, data, num_clusters):
        #TODO
        model = ExpectationMaximization(data )
        assignations = model.clusterize()
        return model, assignations


    def time_series_model(self):
        self.experiment_name = "Time Series Model"
        exp_start_time = time.process_time()
        df, ids, males, females = DatasetLoader.load_temporal_dataset()
        # TODO


    def evolutionary_model(self):
        self.experiment_name = "Evolutionary Model"
        exp_start_time = time.process_time()
        # TODO
