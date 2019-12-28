import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from neupy import algorithms


class ExpectationMaximization:

    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.transformed_input = None
        self.__fit()
        self.__transform()

    def __fit(self):
        print()
        # TODO:

    def __transform(self):
        print()
        # TODO:

    def model_name(self):
        return "EM"

    def clusterize(self):
        print()
        # TODO:
