import pandas
import numpy
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

# Custom imports
from utils.dataset_loader import ParkinsonDataset as PD
from utils.visualizer import *
from sklearn.model_selection import KFold

# EXECUTION_MODE = "RUN"
EXECUTION_MODE = "SEARCH"

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
        components_vec = numpy.arange(5, len(PD.FEATURES) + 1)
        results = pandas.DataFrame(
            columns=['Total-Test', "Total-Params", 'Motor-Test', "Motor-Params"],
            index=components_vec)

        for n_components in components_vec:
            # Dimensionality reduction techniques ____________________________________________________________
            pca = PCA(n_components=n_components, svd_solver='full')
            pca.fit(X_all)
            # Transform dataset to new vector space
            X_all_transformed = pca.transform(X_all - X_all.mean(axis=0))
            if n_components == len(PD.FEATURES):
                print("Original dataset")
                X_all_transformed = X_all
            # GBR Hyper-Parameter search _____________________________________________________________________
            # Define Model, params and grid search scheme with cross validation.
            parameters = {'learning_rate': numpy.linspace(0.0001, 0.005, 5),
                          'max_depth': [8, 10, 15]}
            gbr = GradientBoostingRegressor(loss='ls', n_estimators=20000, n_iter_no_change=10, validation_fraction=0.2)
            clf = GridSearchCV(gbr, parameters, scoring='neg_mean_absolute_error', cv=KFold(n_splits=5, shuffle=True),
                               verbose=1, n_jobs=3)
            # Train two models, one for each target
            for y_target, y_type in zip([y_all_total, y_all_motor], ['Total', 'Motor']):
                print("num-PCs=%d Training %s on %s" % (n_components, model_name, y_type))
                # Perform grid search
                clf.fit(X_all_transformed, y_target)

                # Save results for later processing/analysis ==============================================
                results.at[n_components, y_type + '-Test'] = clf.cv_results_['mean_test_score'][clf.best_index_]
                # results.at[n_components, y_type + '-Train'] = clf.cv_results_['mean_train_score'][clf.best_index_]
                results.at[n_components, y_type + '-Params'] = clf.best_params_
                svr_model = clf.best_estimator_
                print(results)
        results.to_csv("../results/outputs/%s/MAE-diff-components.csv" % model_name)
        print(results)

    # Train best model for total
    # pca = PCA(n_components=5, svd_solver='full')
    # X_train_transformed = pca.fit_transform(X_train)
    # X_test_transformed = pca.transform(X_test - X_test.mean(axis=0))
    # params = {'learning_rate': 0.001, 'loss': 'ls', 'max_depth': 3}
    # pca_gbr_total = GradientBoostingRegressor(n_estimators=2000, **params)
    # pca_gbr_total.fit(X_train_transformed, y_train_total)
    # y_pred_total = pca_gbr_total.predict(X_test_transformed)
    # mae_pca_gbr_total = mean_absolute_error(y_test_total, y_pred_total)
    # _____________________________________________________________________________________________________
    # Experiment on Recursive feature elimination
    params = {'n_iter_no_change': 10,
              'validation_fraction': 0.2,
              'n_estimators': 10000}
    model = GradientBoostingRegressor(learning_rate=0.0042, max_depth=10, **params)

    all_feature_masks, all_mae_log = PD.recursive_feature_elimination(model=model, X=X_all,
                                                                      y_total=y_all_total,
                                                                      y_motor=y_all_motor)
    print("[All] Total features: %s" % numpy.array(PD.FEATURES)[all_feature_masks['Total']])
    print("[All] Motor features: %s" % numpy.array(PD.FEATURES)[all_feature_masks['Motor']])

    X_males = df_males[PD.FEATURES].values
    y_total = df_males[PD.TOTAL_UPDRS].values
    y_motor = df_males[PD.MOTOR_UPDRS].values
    male_feature_masks, male_mae_log = PD.recursive_feature_elimination(model=model, X=X_males,
                                                                        y_total=y_total,
                                                                        y_motor=y_motor)
    print("[Male] Total features: %s" % numpy.array(PD.FEATURES)[male_feature_masks['Total']])
    print("[Male] Motor features: %s" % numpy.array(PD.FEATURES)[male_feature_masks['Motor']])

    X_females = df_females[PD.FEATURES].values
    y_total = df_females[PD.TOTAL_UPDRS].values
    y_motor = df_females[PD.MOTOR_UPDRS].values
    female_feature_masks, female_mae_log = PD.recursive_feature_elimination(model=model, X=X_females,
                                                                            y_total=y_total,
                                                                            y_motor=y_motor)

    print("[Female] Total features: %s" % numpy.array(PD.FEATURES)[female_feature_masks['Total']])
    print("[Female] Motor features: %s" % numpy.array(PD.FEATURES)[female_feature_masks['Motor']])

    plt.figure()
    x_log = range(1, len(all_mae_log['Total']) + 1)
    plt.plot(x_log, all_mae_log['Total'], label="Total UPDRS")
    plt.plot(x_log, all_mae_log['Motor'], label="Motor UPDRS")
    plt.plot(x_log, male_mae_log['Total'], label="Total UPDRS [M]")
    plt.plot(x_log, male_mae_log['Motor'], label="Motor UPDRS [M]")
    plt.plot(x_log, female_mae_log['Total'], label="Total UPDRS [F]")
    plt.plot(x_log, female_mae_log['Motor'], label="Motor UPDRS [F]")
    title = "Recursive Feature Elimination [%s]" % model_name
    plt.title(title)
    plt.xlabel("Number of selected features")
    plt.ylabel("Cross validated average MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../media/" + title.replace(" ", "_") + ".png")
    plt.show()
