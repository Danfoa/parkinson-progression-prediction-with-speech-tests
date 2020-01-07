import pandas
import numpy

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
# Custom imports
from utils.dataset_loader import ParkinsonDataset
from sklearn.model_selection import KFold
from experiments import anfis_model
from clustering_models.em import ExpectationMaximization
from clustering_models.som import SelfOrganizingMap


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


    # Create cross-validation partition
    for algorithm, num_clusters in zip(clustering_algorithms, algorithm_clusters):
        print(algorithm)
        results = pandas.DataFrame(columns=['Total-Test', 'Motor-Test'],
                                   index=clustering_algorithms)

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

                model = anfis_model.MyANFIS(X_train, y_train)
                model.fit(k=0.051, gamma=4000)

                # Save results for later processing/analysis ==============================================
                y_pred = model.predict(X_test)

                y_pred_total_clusters[cluster, :] = y_pred[:, 0]
                # Motor __________________________________________________
                y_pred_motor_clusters[cluster, :] = y_pred[:,1]

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
        results.to_csv(save_path + "[%s]clustering+regression_results.csv" % model_name)
