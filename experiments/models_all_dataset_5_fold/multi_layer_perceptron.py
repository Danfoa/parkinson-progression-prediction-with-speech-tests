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
    # Example of loading the dataset _________________________________________________________________
    df, ids, df_males, df_females = ParkinsonDataset.load_dataset(path="../dataset/parkinsons_updrs.data",
                                                                  return_gender=True)

    # Normalizing/scaling  dataset
    ParkinsonDataset.normalize_dataset(dataset=df, scaler=MinMaxScaler(), inplace=True)
    ParkinsonDataset.normalize_dataset(dataset=df_females, scaler=MinMaxScaler(), inplace=True)
    ParkinsonDataset.normalize_dataset(dataset=df_males, scaler=MinMaxScaler(), inplace=True)

    # Split dataset
    # Used in model cross-validated hyper-parameter search
    X_all = df[ParkinsonDataset.FEATURES].values
    y_all = df[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values

    X_males = df_males[ParkinsonDataset.FEATURES].values
    y_males = df_males[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values

    X_females = df_females[ParkinsonDataset.FEATURES].values
    y_females = df_females[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values

    hidden_units = [500, 400, 300, 200]
    activation = 'sigmoid'
    lr = 0.0005

    results = pandas.DataFrame(columns=['Total-Test', 'Motor-Test'],
                               index=["All", "Males", "Females"])

    # Create CV loop, providing indexes of training and testing
    total_results, motor_results = [], []
    cv_splitter = KFold(n_splits=5, shuffle=True)

    # ALL
    for train_index, test_index in cv_splitter.split(X_all):
        model = keras.Sequential()
        for layer in range(len(hidden_units)):
            model.add(layers.Dense(units=hidden_units[layer], activation=activation))
        # 2 units in the output layer (Total and Motor)
        model.add(layers.Dense(units=2))
        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        print(X_all[train_index].shape)
        print(X_all[test_index].shape)
        history = model.fit(x=X_all[train_index],
                            y=y_all[train_index],
                            epochs=1000,
                            validation_split=0.1,
                            shuffle=True,
                            verbose=0,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])

        y_pred = model.predict(X_all[test_index])

        mae_total = mean_absolute_error(y_all[test_index, 0], y_pred[:, 0])
        mae_motor = mean_absolute_error(y_all[test_index, 1], y_pred[:, 1])
        total_results.append(mae_total)
        motor_results.append(mae_motor)

    results.at["All", "Total-Test"] = total_results
    results.at["All", "Motor-Test"] = motor_results
    print(results)

    # MALE
    total_results, motor_results = [], []
    for train_index, test_index in cv_splitter.split(X_males):
        model = keras.Sequential()
        for layer in range(len(hidden_units)):
            model.add(layers.Dense(units=hidden_units[layer], activation=activation))
        # 2 units in the output layer (Total and Motor)
        model.add(layers.Dense(units=2))
        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        print(X_males[train_index].shape)
        print(X_males[test_index].shape)
        history = model.fit(x=X_males[train_index],
                            y=y_males[train_index],
                            epochs=1000,
                            validation_split=0.1,
                            shuffle=True,
                            verbose=0,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])

        y_pred = model.predict(X_males[test_index])

        mae_total = mean_absolute_error(y_males[test_index, 0], y_pred[:, 0])
        mae_motor = mean_absolute_error(y_males[test_index, 1], y_pred[:, 1])
        total_results.append(mae_total)
        motor_results.append(mae_motor)

    results.at["Males", "Total-Test"] = total_results
    results.at["Males", "Motor-Test"] = motor_results
    print(results)

    # FEMALE
    total_results, motor_results = [], []
    for train_index, test_index in cv_splitter.split(X_females):
        model = keras.Sequential()
        for layer in range(len(hidden_units)):
            model.add(layers.Dense(units=hidden_units[layer], activation=activation))
        # 2 units in the output layer (Total and Motor)
        model.add(layers.Dense(units=2))
        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        print(X_females[train_index].shape)
        print(X_females[test_index].shape)
        history = model.fit(x=X_females[train_index],
                            y=y_females[train_index],
                            epochs=1000,
                            validation_split=0.1,
                            shuffle=True,
                            verbose=0,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])

        y_pred = model.predict(X_females[test_index])

        mae_total = mean_absolute_error(y_females[test_index, 0], y_pred[:, 0])
        mae_motor = mean_absolute_error(y_females[test_index, 1], y_pred[:, 1])
        total_results.append(mae_total)
        motor_results.append(mae_motor)

    results.at["Females", "Total-Test"] = total_results
    results.at["Females", "Motor-Test"] = motor_results
    print(results)

    results.to_csv("../../results/outputs/%s/MAE-final-%s-KFold.csv" % (model_name, model_name))