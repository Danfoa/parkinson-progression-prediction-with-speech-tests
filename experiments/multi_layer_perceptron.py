import pandas
import numpy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error

# Custom imports
from utils.dataset_loader import ParkinsonDataset
from utils.visualizer import *

if __name__ == '__main__':
    model_name = "MLP"
    # Example of loading the dataset _________________________________________________________________
    df = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                       return_gender=False)
    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=RobustScaler(),
                                                             inplace=True)
    # Split dataset
    # Used in model cross-validated hyper-parameter search
    X_all = df[ParkinsonDataset.FEATURES].values
    y_all_total = df[ParkinsonDataset.TOTAL_UPDRS].values
    y_all_motor = df[ParkinsonDataset.MOTOR_UPDRS].values
    # Use for evaluation selected model
    X_train, X_test, y_train, y_test = ParkinsonDataset.split_dataset(dataset=df,
                                                                      subject_partitioning=False)
    # Get TOTAL UPDRS targets
    y_train_total, y_test_total = y_train[:, 0], y_test[:, 0]
    # Get MOTOR UPDRS targets
    y_train_motor, y_test_motor = y_train[:, 1], y_test[:, 1]
    # ________________________________________________________________________________________________

    hidden_units = [2000, 1000, 500, 250, 100]
    hidden_layers = [1, 2, 3, 4, 5]

    results = pandas.DataFrame(
        columns=['Total-Test', "Total-Params", 'Motor-Test', "Motor-Params"],
        index = hidden_layers)

    # Find best MLP model
    for num_layers in hidden_layers:
        best_total = None
        best_motor = None
        for activation in ['relu', 'sigmoid', 'softmax']:
            for lr in numpy.linspace(0.0001, 0.1, 10):
                model = keras.Sequential()
                for layer in range(num_layers):
                    model.add(layers.Dense(units=hidden_units[layer], activation=activation))
                    model.add(layers.Dropout(0.5))

                # 2 units in the output layer (Total and Motor)
                model.add(layers.Dense(units=2))
                optimizer = tf.keras.optimizers.Adam(lr)
                model.compile(loss='mae',
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

                if best_total is None or abs(mae_total) < abs(best_total):
                    best_total = mae_total
                    results.at[num_layers, 'Total-Test'] = mae_total
                    results.at[num_layers, 'Total-Params'] = {'activation': activation, 'lr': lr}

                if best_motor is None or abs(mae_motor) < abs(best_motor):
                    best_motor = mae_motor
                    results.at[num_layers, 'Motor-Test'] = mae_motor
                    results.at[num_layers, 'Motor-Params'] = {'activation': activation, 'lr': lr}

                print(results)
    results.to_csv("../results/outputs/%s/MAE-diff-num-layers.csv" % model_name)
    print(results)

