import pandas
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# Custom imports
from utils.dataset_loader import ParkinsonDataset

if __name__ == '__main__':
    model_name = "SVR"
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
    y_all_total = df[ParkinsonDataset.TOTAL_UPDRS].values
    y_all_motor = df[ParkinsonDataset.MOTOR_UPDRS].values

    X_males = df_males[ParkinsonDataset.FEATURES].values
    y_males = df_males[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values
    y_males_total = df_males[ParkinsonDataset.TOTAL_UPDRS].values
    y_males_motor = df_males[ParkinsonDataset.MOTOR_UPDRS].values

    X_females = df_females[ParkinsonDataset.FEATURES].values
    y_females = df_females[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values
    y_females_total = df_females[ParkinsonDataset.TOTAL_UPDRS].values
    y_females_motor = df_females[ParkinsonDataset.MOTOR_UPDRS].values

    cv_splitter = KFold(n_splits=5, shuffle=True)
    results = pandas.DataFrame(columns=['Total-Test', 'Motor-Test'],
                               index=["All", "Males", "Females"])

    # Experiment
    params = {'kernel': 'rbf',
              'C': 10,
              'gamma': 1}

    # ALL
    total_results, motor_results = [], []
    for i in range(5):
        X_train, X_test, y_train, y_test = ParkinsonDataset.split_dataset(dataset=df,
                                                                          subject_partitioning=False)
        # Get TOTAL UPDRS targets
        y_train_total, y_test_total = y_train[:, 0], y_test[:, 0]
        # Get MOTOR UPDRS targets
        y_train_motor, y_test_motor = y_train[:, 1], y_test[:, 1]

        model = SVR(**params)
        # Total __________________________________________________
        model.fit(X_train, y_train_total)
        y_pred_total = model.predict(X_test)
        # Motor __________________________________________________
        model.fit(X_train, y_train_motor)
        y_pred_motor = model.predict(X_test)

        mae_total = mean_absolute_error(y_test_total, y_pred_total)
        mae_motor = mean_absolute_error(y_test_motor, y_pred_motor)
        total_results.append(mae_total)
        motor_results.append(mae_motor)

    results.at["All", "Total-Test"] = total_results
    results.at["All", "Motor-Test"] = motor_results
    print(results)

    # MALES
    total_results, motor_results = [], []
    for i in range(5):
        X_train, X_test, y_train, y_test = ParkinsonDataset.split_dataset(dataset=df_males,
                                                                          subject_partitioning=False)
        # Get TOTAL UPDRS targets
        y_train_total, y_test_total = y_train[:, 0], y_test[:, 0]
        # Get MOTOR UPDRS targets
        y_train_motor, y_test_motor = y_train[:, 1], y_test[:, 1]

        model = SVR(**params)
        # Total __________________________________________________
        model.fit(X_train, y_train_total)
        y_pred_total = model.predict(X_test)
        # Motor __________________________________________________
        model.fit(X_train, y_train_motor)
        y_pred_motor = model.predict(X_test)

        mae_total = mean_absolute_error(y_test_total, y_pred_total)
        mae_motor = mean_absolute_error(y_test_motor, y_pred_motor)
        total_results.append(mae_total)
        motor_results.append(mae_motor)

    results.at["Males", "Total-Test"] = total_results
    results.at["Males", "Motor-Test"] = motor_results
    print(results)

    # FEMALE
    total_results, motor_results = [], []
    for i in range(5):
        X_train, X_test, y_train, y_test = ParkinsonDataset.split_dataset(dataset=df_females,
                                                                          subject_partitioning=False)
        # Get TOTAL UPDRS targets
        y_train_total, y_test_total = y_train[:, 0], y_test[:, 0]
        # Get MOTOR UPDRS targets
        y_train_motor, y_test_motor = y_train[:, 1], y_test[:, 1]

        model = SVR(**params)
        # Total __________________________________________________
        model.fit(X_train, y_train_total)
        y_pred_total = model.predict(X_test)
        # Motor __________________________________________________
        model.fit(X_train, y_train_motor)
        y_pred_motor = model.predict(X_test)

        mae_total = mean_absolute_error(y_test_total, y_pred_total)
        mae_motor = mean_absolute_error(y_test_motor, y_pred_motor)
        total_results.append(mae_total)
        motor_results.append(mae_motor)

    results.at["Females", "Total-Test"] = total_results
    results.at["Females", "Motor-Test"] = motor_results
    print(results)

    results.to_csv("../../results/outputs/%s/MAE-final-%s-KFold.csv" % (model_name, model_name))
