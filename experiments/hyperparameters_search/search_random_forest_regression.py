import pandas
import numpy
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold

# Custom imports
from utils.dataset_loader import ParkinsonDataset
from utils.models_all_dataset_plot import *

if __name__ == '__main__':
    model_name = "RFR"
    # Example of loading the dataset _________________________________________________________________
    df = ParkinsonDataset.load_dataset(path="../dataset/parkinsons_updrs.data", return_gender=False)

    # Normalizing/scaling  dataset
    ParkinsonDataset.normalize_dataset(dataset=df, scaler=MinMaxScaler(), inplace=True)

    # Split dataset
    # Used in model cross-validated hyper-parameter search
    X_all = df[ParkinsonDataset.FEATURES].values
    y_all = df[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values
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

    components_vec = numpy.array([4, 6, 10, 14, len(ParkinsonDataset.FEATURES)])
    results = pandas.DataFrame(
        columns=['Total-Test', "Total-Params", 'Motor-Test', "Motor-Params"],
        index=components_vec)

    for n_components in components_vec:
        # Dimensionality reduction techniques ____________________________________________________________
        pca = PCA(n_components=n_components, svd_solver='full')
        pca.fit(X_all)
        # Transform dataset to new vector space
        X_all_transformed = pca.transform(X_all - X_all.mean(axis=0))
        if n_components == len(ParkinsonDataset.FEATURES):
            print("Original dataset")
            X_all_transformed = X_all
        # X_train_transformed = pca.transform(X_train - X_train.mean(axis=0))
        # X_test_transformed = pca.transform(X_test - X_test.mean(axis=0))

        # SVR Hyper-Parameter search _____________________________________________________________________
        # Define Model, params and grid search scheme with cross validation.
        parameters = {'n_estimators': [150, 200, 500]}
        rfr = RandomForestRegressor(criterion='mae')
        clf = GridSearchCV(rfr, parameters, scoring='neg_mean_absolute_error', cv=KFold(n_splits=5, shuffle=True),
                           verbose=1, n_jobs=4)
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
    results.to_csv("../../results/outputs/%s/MAE-diff-components.csv" % model_name)
    print(results)


