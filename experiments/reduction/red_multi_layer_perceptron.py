import pandas
import numpy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Custom imports
from utils.dataset_loader import ParkinsonDataset

if __name__ == '__main__':
    model_name = "MLP_Reduced_Dataset"
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

                hidden_units = [1000, 500, 300]
                activation = 'relu'
                lr = 0.0001

                model = keras.Sequential()
                model.add(layers.Dense(units=hidden_units[0], activation=activation))
                model.add(layers.Dropout(0.5))
                model.add(layers.Dense(units=hidden_units[1], activation=activation))
                model.add(layers.Dropout(0.5))
                model.add(layers.Dense(units=hidden_units[2], activation=activation))
                model.add(layers.Dropout(0.5))

                # 2 units in the output layer (Total and Motor)
                model.add(layers.Dense(units=2))
                optimizer = tf.keras.optimizers.Adam(lr)
                model.compile(loss='mse',
                              optimizer=optimizer,
                              metrics=['mae', 'mse'])

                history = model.fit(x=X_train,
                                    y=y_train,
                                    epochs=500,
                                    validation_data=(X_test, y_test),
                                    shuffle=True,
                                    verbose=0,
                                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])

                y_pred = model.predict(X_test)

                mae_total = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
                mae_motor = mean_absolute_error(y_test[:, 1], y_pred[:, 1])


                results.at[k, 'Total-Test'] = mae_total
                results.at[k, 'Total-Params'] = {'activation': activation, 'lr': lr}
                results.at[k, 'Motor-Test'] = mae_motor
                results.at[k, 'Motor-Params'] = {'activation': activation, 'lr': lr}

            results.to_csv(save_path + model_name + "/" + algorithm + '/MAE-C=%d-diff-k.csv' % c)
            print(results)

