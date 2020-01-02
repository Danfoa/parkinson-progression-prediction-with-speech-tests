import pandas
import numpy
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

# Custom imports
from utils.dataset_loader import ParkinsonDataset as PD
from utils.visualizer import *

EXECUTION_MODE = "RUN"
# EXECUTION_MODE = "SEARCH"

if __name__ == '__main__':
    model_name = "GBR"
    # Example of loading the dataset _________________________________________________________________
    df, ids, df_males, df_females = PD.load_dataset(path="dataset/parkinsons_updrs.data", return_gender=True)
    PD.normalize_dataset(dataset=df, scaler=MinMaxScaler(), inplace=True)
    PD.normalize_dataset(dataset=df_females, scaler=MinMaxScaler(), inplace=True)
    PD.normalize_dataset(dataset=df_males, scaler=MinMaxScaler(), inplace=True)

    # Normalizing/scaling  dataset
    feature_normalizers = PD.normalize_dataset(dataset=df,
                                               scaler=MinMaxScaler(),
                                               inplace=True)
    # Split dataset
    # Used in model cross-validated hyper-parameter search
    X_all = df[PD.FEATURES].values
    y_all_total = df[PD.TOTAL_UPDRS].values
    y_all_motor = df[PD.MOTOR_UPDRS].values
    # Use for evaluation selected model
    X_train, X_test, y_train, y_test = PD.split_dataset(dataset=df,
                                                        subject_partitioning=False)
    # Get TOTAL UPDRS targets
    y_train_total, y_test_total = y_train[:, 0], y_test[:, 0]
    # Get MOTOR UPDRS targets
    y_train_motor, y_test_motor = y_train[:, 1], y_test[:, 1]
    # ________________________________________________________________________________________________

    # Design experiment to train model hyper-parameters:
    if EXECUTION_MODE == "SEARCH":
        components_vec = numpy.arange(4, 16)

        # GBR Hyper-Parameter search _____________________________________________________________________
        # Define Model, params and grid search scheme with cross validation.

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(10, 12))
        for lr in numpy.linspace(0.0001, 0.005, 7):
            for md in [6, 8, 10, 15]:
                gbr = GradientBoostingRegressor(learning_rate=lr, max_depth=md, loss='ls',
                                                n_estimators=20000, n_iter_no_change=10,
                                                validation_fraction=0.2)
                # Train two models, one for each target
                for y_target, y_test_target, y_type in zip([y_train_total, y_train_motor],
                                                           [y_test_total, y_test_motor],
                                                           ['Total', 'Motor']):
                    # Perform grid search
                    gbr.fit(X_train, y_target)
                    y_pred = gbr.predict(X_test)
                    test_score = mean_absolute_error(y_pred=y_pred, y_true=y_test_target)
                    sel_ax = ax1 if y_type == "Total" else ax2
                    sel_ax.plot(numpy.arange(0, len(gbr.train_score_)), gbr.train_score_,
                                label="MAE:%.5f - [lr:%.7f, md:%d]" % (test_score, lr, md))
                    print("Training %s on %s [lr:%.7f, md:%d] - MAE:%.5f" % (model_name, y_type, lr, md, test_score))


        plt.xlabel('Boosting Iterations')
        ax1.set_ylabel('Training MSE')
        ax1.set_title("Total UPDRS")
        ax2.set_ylabel('Training MSE')
        ax2.set_title("Motor UPDRS")
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.savefig("../media/GBR_grid_search.png")
        plt.show()

    # _____________________________________________________________________________________________________
    # Experiment on Recursive feature elimination
    params = {'n_iter_no_change': 10,
              'validation_fraction': 0.2,
              'n_estimators': 10000}
    gbr_total = GradientBoostingRegressor(learning_rate=0.0050, max_depth=8, **params)
    gbr_motor = GradientBoostingRegressor(learning_rate=0.0042, max_depth=10, **params)

    features = PD.FEATURES
    features_vec = numpy.array(features)
    best_features = numpy.array(features)
    nof_list = numpy.arange(2, len(features), dtype=numpy.short)
    high_score = 0
    # Variable to store the optimum features
    nof = 0
    score_total = []
    score_motor = []

    for n in range(len(nof_list)):
        for y_target, y_test_target, y_type in zip([y_train_total, y_train_motor],
                                                   [y_test_total, y_test_motor],
                                                   ['Total', 'Motor']):
            model = gbr_total if y_type == "Total" else gbr_motor
            rfe = RFE(model, nof_list[n])
            # Filter out features from dataset
            X_train_rfe = rfe.fit_transform(X_train, y_target)
            X_test_rfe = rfe.transform(X_test)
            # Predict gbr outcome with the selected features
            model.fit(X_train_rfe, y_target)
            y_test_pred = model.predict(X_test_rfe)
            # Evaluate model performance
            test_score = mean_absolute_error(y_pred=y_test_pred, y_true=y_test_target)
            print("[%s] Checking performance with %d features MAE:%.5f" % (y_type, nof_list[n], test_score))
            print(features_vec[rfe.get_support(indices=True)])
            if y_type == "Total":
                score_total.append(test_score)
            else:
                score_motor.append(test_score)

            sel_features_ids = rfe.get_support(indices=True)

    plt.figure()
    plt.plot(nof_list, score_total, '-o', label="Total UPDRS")
    plt.plot(nof_list, score_motor, '-o', label="Motor UPDRS")
    plt.title("Gradient Boosting Regression RFE")
    plt.ylabel("MAE")
    plt.xlabel("Size of feature set")
    plt.grid()
    plt.legend()
    plt.savefig("../media/Recursive_Feature_Elimination_[GBR].png")
    plt.show()

    results = pandas.DataFrame(columns=[""])