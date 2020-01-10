import matplotlib.pyplot as plt
import numpy
from sklearn.ensemble import GradientBoostingRegressor
# Sklearn imports
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from utils.dataset_loader import ParkinsonDataset as PD

if __name__ == '__main__':
    model_name = "GBR"
    # Example of loading the dataset _________________________________________________________________
    df, ids, df_males, df_females = PD.load_dataset(path="../dataset/parkinsons_updrs.data", return_gender=True)
    PD.normalize_dataset(dataset=df, scaler=MinMaxScaler(), inplace=True)
    PD.normalize_dataset(dataset=df_females, scaler=MinMaxScaler(), inplace=True)
    PD.normalize_dataset(dataset=df_males, scaler=MinMaxScaler(), inplace=True)

    # Normalizing/scaling  dataset
    feature_normalizers = PD.normalize_dataset(dataset=df,
                                               scaler=MinMaxScaler(),
                                               inplace=True)
    # Split dataset
    X_all = df[PD.FEATURES].values
    y_all_total = df[PD.TOTAL_UPDRS].values
    y_all_motor = df[PD.MOTOR_UPDRS].values

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
    plt.savefig("../../media/rfe/" + title.replace(" ", "_") + ".png")
    plt.show()
