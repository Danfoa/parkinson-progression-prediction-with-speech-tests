import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
# Sklearn imports
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from utils.dataset_loader import ParkinsonDataset

if __name__ == '__main__':
    model_name = "RFR"
    # Example of loading the dataset _________________________________________________________________
    df, ids, df_males, df_females = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                                                  return_gender=True)
    ParkinsonDataset.normalize_dataset(dataset=df, scaler=MinMaxScaler(), inplace=True)
    ParkinsonDataset.normalize_dataset(dataset=df_females, scaler=MinMaxScaler(), inplace=True)
    ParkinsonDataset.normalize_dataset(dataset=df_males, scaler=MinMaxScaler(), inplace=True)

    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=MinMaxScaler(),
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
    # Experiment on Recursive feature elimination
    params = {'n_estimators': 500}
    model = RandomForestRegressor(**params)

    all_feature_masks, all_mae_log = ParkinsonDataset.recursive_feature_elimination(model=model, X=X_all,
                                                                                    y_total=y_all_total,
                                                                                    y_motor=y_all_motor)
    print("[All] Total features: %s" % numpy.array(ParkinsonDataset.FEATURES)[all_feature_masks['Total']])
    print("[All] Motor features: %s" % numpy.array(ParkinsonDataset.FEATURES)[all_feature_masks['Motor']])

    X_males = df_males[ParkinsonDataset.FEATURES].values
    y_total = df_males[ParkinsonDataset.TOTAL_UPDRS].values
    y_motor = df_males[ParkinsonDataset.MOTOR_UPDRS].values
    male_feature_masks, male_mae_log = ParkinsonDataset.recursive_feature_elimination(model=model, X=X_males,
                                                                                      y_total=y_total,
                                                                                      y_motor=y_motor)
    print("[Male] Total features: %s" % numpy.array(ParkinsonDataset.FEATURES)[male_feature_masks['Total']])
    print("[Male] Motor features: %s" % numpy.array(ParkinsonDataset.FEATURES)[male_feature_masks['Motor']])

    X_females = df_females[ParkinsonDataset.FEATURES].values
    y_total = df_females[ParkinsonDataset.TOTAL_UPDRS].values
    y_motor = df_females[ParkinsonDataset.MOTOR_UPDRS].values
    female_feature_masks, female_mae_log = ParkinsonDataset.recursive_feature_elimination(model=model, X=X_females,
                                                                                          y_total=y_total,
                                                                                          y_motor=y_motor)

    print("[Female] Total features: %s" % numpy.array(ParkinsonDataset.FEATURES)[female_feature_masks['Total']])
    print("[Female] Motor features: %s" % numpy.array(ParkinsonDataset.FEATURES)[female_feature_masks['Motor']])

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
    plt.savefig("../media/rfe/" + title.replace(" ", "_") + ".png")
    plt.show()
