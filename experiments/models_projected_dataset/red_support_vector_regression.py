import pandas
import numpy

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
# Custom imports
from utils.dataset_loader import ParkinsonDataset
from sklearn.model_selection import KFold

if __name__ == '__main__':
    model_name = "SVR"
    load_path = "../../results/reduction/"
    save_path = "../../results/outputs/"

    # clustering algorithms to test
    clustering_algorithms = ["fuzzy_c_means", "som", "em"]
    # Number of algorithm_clusters to test in each algorithm
    algorithm_clusters = [5, 9, 12]

    df = ParkinsonDataset.load_dataset(path="../dataset/parkinsons_updrs.data",
                                       return_gender=False)
    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=StandardScaler(),
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

            # Allocate matrix for predictions
            y_pred_total_clusters = numpy.ones((num_clusters, len(test_index)))
            y_pred_motor_clusters = numpy.ones((num_clusters, len(test_index)))
            # Iterate through each of the projected data-sets and predict a result
            for cluster in range(num_clusters):
                X_projected = numpy.load(load_path + algorithm + '/C=%d-K=%d-reduced-dataset-std-scaler.npy' % (num_clusters,
                                                                                                     cluster))
                X_train, X_test = X_projected[train_index, :], X_projected[test_index, :]

                # SVR
                model = SVR(kernel='rbf', C=10, gamma=1)

                # Total __________________________________________________
                model.fit(X_train, y_total_train)
                y_pred_total_clusters[cluster, :] = model.predict(X_test)
                # Motor __________________________________________________
                model.fit(X_train, y_motor_train)
                y_pred_motor_clusters[cluster, :] = model.predict(X_test)

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


