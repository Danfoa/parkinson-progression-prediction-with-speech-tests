import numpy
import pandas
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Custom imports
from utils.dataset_loader import ParkinsonDataset

if __name__ == '__main__':
    model_name = "MLP"
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
                                                             scaler=MinMaxScaler(),
                                                             inplace=True)
    X_all = df[ParkinsonDataset.FEATURES].values
    y_all = df[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values
    y_total = df[ParkinsonDataset.TOTAL_UPDRS].values
    y_motor = df[ParkinsonDataset.MOTOR_UPDRS].values

    indices = numpy.arange(len(y_total))

    results = pandas.DataFrame(columns=['Total-Test', 'Motor-Test'],
                               index=clustering_algorithms)

    # Create cross-validation partition
    for algorithm, num_clusters in zip(clustering_algorithms, algorithm_clusters):
        # Create CV loop, providing indexes of training and testing
        total_results, motor_results = [], []

        K = 5
        for i in range(K):
            X_train, X_test, y_train, y_test, train_index, test_index = ParkinsonDataset.split_dataset_indices(dataset=df, indices=indices)
            y_total_test = y_test[:, 0]
            y_motor_test = y_test[:, 1]
            # Allocate matrix for predictions
            y_pred_total_clusters = numpy.ones((num_clusters, len(test_index)))
            y_pred_motor_clusters = numpy.ones((num_clusters, len(test_index)))
            # Iterate through each of the projected data-sets and predict a result
            for cluster in range(num_clusters):
                X_projected = numpy.load(load_path + algorithm + '/C=%d-K=%d-reduced-dataset.npy' % (num_clusters,
                                                                                                     cluster))
                X_train, X_test = X_projected[train_index, :], X_projected[test_index, :]

                hidden_units = [500, 400, 300, 200]
                activation = 'sigmoid'
                lr = 0.0005

                model = keras.Sequential()
                for layer in range(len(hidden_units)):
                    model.add(layers.Dense(units=hidden_units[layer], activation=activation))
                # 2 units in the output layer (Total and Motor)
                model.add(layers.Dense(units=2))
                optimizer = tf.keras.optimizers.Adam(lr)
                model.compile(loss='mse',
                              optimizer=optimizer,
                              metrics=['mae', 'mse'])

                history = model.fit(x=X_train,
                                    y=y_train,
                                    epochs=1000,
                                    validation_split=0.1,
                                    shuffle=True,
                                    verbose=0,
                                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])

                y_pred = model.predict(X_test)

                y_pred_total_clusters[cluster, :] = y_pred[:, 0]
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
