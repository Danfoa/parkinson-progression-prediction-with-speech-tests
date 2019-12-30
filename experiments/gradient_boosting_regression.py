import pandas
import numpy

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

# Custom imports
from utils.dataset_loader import ParkinsonDataset
from utils.visualizer import *

if __name__ == '__main__':
    model = "GBR"
    # Example of loading the dataset _________________________________________________________________
    df = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                       return_gender=False)
    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=StandardScaler(),
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

    # Design experiment to train model hyper-parameters:
    components_vec = numpy.arange(4, 12)
    results = pandas.DataFrame(
        columns=['Total-Test', "Total-Params", 'Motor-Test', "Motor-Params"],
        index=components_vec)

    for n_components in components_vec:
        # Dimensionality reduction techniques ____________________________________________________________
        pca = PCA(n_components=n_components, svd_solver='full')
        pca.fit(X_all)
        # Transform dataset to new vector space
        X_all_transformed = pca.transform(X_all - X_all.mean(axis=0))
        # X_train_transformed = pca.transform(X_train - X_train.mean(axis=0))
        # X_test_transformed = pca.transform(X_test - X_test.mean(axis=0))

        # SVR Hyper-Parameter search _____________________________________________________________________
        # Define Model, params and grid search scheme with cross validation.
        parameters = {'loss': ['ls', 'lad'],
                      'learning_rate': [0.001, 0.01, 0.1, 0.5],
                      'max_depth': [3, 4, 5, 6]}
        svr = GradientBoostingRegressor(n_estimators=1500)
        clf = GridSearchCV(svr, parameters, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=3)
        # Train two models, one for each target
        for y_target, y_type in zip([y_all_total, y_all_motor], ['Total', 'Motor']):
            print("num-PCs=%d Training %s on %s" % (n_components, model, y_type))
            # Perform grid search
            clf.fit(X_all_transformed, y_target)

            # Save results for later processing/analysis ==============================================
            results.at[n_components, y_type + '-Test'] = clf.cv_results_['mean_test_score'][clf.best_index_]
            # results.at[n_components, y_type + '-Train'] = clf.cv_results_['mean_train_score'][clf.best_index_]
            results.at[n_components, y_type + '-Params'] = clf.best_params_
            svr_model = clf.best_estimator_
            print(results)
    results.to_csv("../results/outputs/%s/MAE-diff-components.csv" % model)
    print(results)

