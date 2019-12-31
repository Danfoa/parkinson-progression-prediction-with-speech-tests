import pandas
import numpy

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# Custom imports
from utils.dataset_loader import ParkinsonDataset

if __name__ == '__main__':
    model = "SVR_Reduced_Dataset"
    load_path = "../../results/reduction/"
    save_path = "../../results/outputs/"

    clustering_algorithms = ["fuzzy_c_means", "som", "em"]
    clusters = [[4, 5], [9], [12]]

    df = ParkinsonDataset.load_dataset(path="../dataset/parkinsons_updrs.data",
                                       return_gender=False)
    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=StandardScaler(),
                                                             inplace=True)

    y_all_total = df[ParkinsonDataset.TOTAL_UPDRS].values
    y_all_motor = df[ParkinsonDataset.MOTOR_UPDRS].values

    for i in range(len(clustering_algorithms)):
        algorithm = clustering_algorithms[i]
        for c in clusters[i]:
            # Design experiment to train model hyper-parameters:
            clusters_vec = numpy.arange(0, c)
            results = pandas.DataFrame(
                columns=['Total-Test', "Total-Params", 'Motor-Test', "Motor-Params"],
                index=clusters_vec)

            for k in range(c):
                X = numpy.load(load_path + algorithm + '/C=%d-K=%d-reduced-dataset.npy' % (c, k))

                # Use for evaluation selected model
                X_train, X_test, y_train, y_test = ParkinsonDataset.split_reduced_dataset(X=X, dataset=df,
                                                                                    subject_partitioning=False)
                # Get TOTAL UPDRS targets
                y_train_total, y_test_total = y_train[:, 0], y_test[:, 0]
                # Get MOTOR UPDRS targets
                y_train_motor, y_test_motor = y_train[:, 1], y_test[:, 1]
                # ________________________________________________________________________________________________

                # SVR
                svr = SVR(kernel='rbf', C=100, gamma=500)

                # Train two models, one for each target
                for y_target_train, y_target_test, y_type in zip([y_train_total, y_train_motor], [y_test_total, y_test_motor],
                                                                 ['Total', 'Motor']):
                    print("algorithm=%s C=%d K=%d Training %s on %s" % (algorithm, c, k, model, y_type))
                    # Perform grid search
                    svr.fit(X_train, y_target_train)

                    # Save results for later processing/analysis ==============================================
                    y_pred = svr.predict(X_test)

                    results.at[k, y_type + '-Test'] = mean_absolute_error(y_pred, y_target_test)
                    results.at[k, y_type + '-Params'] = svr.get_params()
                    print(results)

            results.to_csv(save_path + model + "/" + algorithm + '/MAE-C=%d-diff-k.csv' %c)
            print(results)

