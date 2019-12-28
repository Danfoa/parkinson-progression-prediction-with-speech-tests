from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.transformed_input = None
        self.__fit_transform()

    def __fit_transform(self):
        clf = LinearDiscriminantAnalysis()
        # clf.fit_transform(X, y)
        print()
        # TODO:

    def model_name(self):
        return "LDA"

    def clusterize(self):
        print()
        # TODO:
