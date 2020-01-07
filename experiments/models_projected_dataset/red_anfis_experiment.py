import pandas
import numpy

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
# Custom imports
from utils.dataset_loader import ParkinsonDataset
from sklearn.model_selection import KFold

import experiments.anfis_packages.anfis
from experiments.anfis_packages import membershipfunction
from experiments.anfis_packages.anfis import ANFIS


class MyANFIS:
    def __init__(self, data, y):
        self.data = data
        self.y = y
        self.num_invars = numpy.size(data, 1)

        mf = self.create_gauss_mfs(data)
        self.mfc = membershipfunction.MemFuncs(mf)
        self.anf = None

    def fit(self, k, gamma):
        self.anf = ANFIS(self.data, self.y, self.mfc)
        self.anf.trainHybridJangOffLine(k, initialGamma=gamma, epochs=10)

    def predict(self, data):
        return experiments.anfis_packages.anfis.predict(self.anf, data)

    def make_gauss_mfs(self, sigma, mu_list):
        '''Return a list of gaussian mfs, same sigma, list of means'''
        l = []
        for mu in mu_list:
            l.append(['gaussmf', {'mean': mu, 'sigma': sigma}])
        # mf = [GaussMembFunc(mu, sigma) for mu in mu_list]
        return l

    def create_gauss_mfs(self, x, num_mfs=2, num_out=2, hybrid=True):
        minvals = numpy.min(self.data, axis=0)

        maxvals = numpy.max(self.data, axis=0)
        # ranges = np.abs(maxvals) - np.abs(minvals)
        ranges = numpy.absolute(maxvals - minvals)
        invars = []
        for i in range(self.num_invars):
            sigma = ranges[i] / num_mfs
            mulist = numpy.linspace(minvals[i], maxvals[i], num_mfs).tolist()
            invars.append(self.make_gauss_mfs(sigma, mulist))
        return invars


class GaussMembFunc:
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        val = numpy.exp(-numpy.pow(x - self.mu, 2) / (2 * self.sigma ** 2))
        return val


if __name__ == '__main__':
    model_name = "ANFIS"
    load_path = "../../results/reduction/"
    save_path = "../../results/outputs/"

    # clustering algorithms to test
    clustering_algorithms = ["fuzzy_c_means", "som", "em"]
    # Number of algorithm_clusters to test in each algorithm
    algorithm_clusters = [4, 4, 4]

    df = ParkinsonDataset.load_dataset(path="../dataset/parkinsons_updrs.data",
                                       return_gender=False)
    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=MinMaxScaler(),
                                                             inplace=True)
    X_all = df[ParkinsonDataset.FEATURES].values
    y_total = df[ParkinsonDataset.TOTAL_UPDRS].values
    y_motor = df[ParkinsonDataset.MOTOR_UPDRS].values

    results = pandas.DataFrame(columns=['Total-Test', 'Motor-Test'],
                               index=clustering_algorithms)

    # Create cross-validation partition
    for algorithm, num_clusters in zip(clustering_algorithms, algorithm_clusters):
        # Create CV loop, providing indexes of training and testing
        total_results, motor_results = [], []
        cv_splitter = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in cv_splitter.split(X_all):
            # Get ground truth
            y_total_train, y_total_test = y_total[train_index], y_total[test_index]
            y_motor_train, y_motor_test = y_motor[train_index], y_motor[test_index]

            y_train = numpy.concatenate((y_total_train.reshape(-1, 1), y_motor_train.reshape(-1, 1)), axis=1)

            # Allocate matrix for predictions
            y_pred_total_clusters = numpy.ones((num_clusters, len(test_index)))
            y_pred_motor_clusters = numpy.ones((num_clusters, len(test_index)))
            # Iterate through each of the projected data-sets and predict a result
            for cluster in range(num_clusters):
                X_projected = numpy.load(load_path + algorithm + '/C=%d-K=%d-reduced-dataset.npy' % (num_clusters,
                                                                                                     cluster))

                X_train, X_test = X_projected[train_index, :], X_projected[test_index, :]

                model = MyANFIS(X_train, y_train)
                model.fit(k=0.051, gamma=4000)

                # Save results for later processing/analysis ==============================================
                y_pred = model.predict(X_test)

                y_pred_total_clusters[cluster, :] = y_pred[:, 0]
                # Motor __________________________________________________
                y_pred_motor_clusters[cluster, :] = y_pred[:, 1]

            y_ensembled_total = y_pred_total_clusters.sum(axis=0) / num_clusters
            y_ensembled_motor = y_pred_motor_clusters.sum(axis=0) / num_clusters
            # Get results from current fold
            fold_total_MAE = mean_absolute_error(y_true=y_total_test, y_pred=y_ensembled_total)
            fold_motor_MAE = mean_absolute_error(y_true=y_motor_test, y_pred=y_ensembled_motor)
            # Save fold value
            total_results.append(fold_total_MAE)
            motor_results.append(fold_motor_MAE)

        results.at[algorithm, "Total-Test"] = total_results
        results.at[algorithm, "Motor-Test"] = motor_results
        print(results)
    results.to_csv(save_path + "%s/MAE-clustering+regression_results.csv" % model_name)
