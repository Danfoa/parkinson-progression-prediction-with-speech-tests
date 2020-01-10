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
from utils.dataset_loader import ParkinsonDataset as PD

if __name__ == '__main__':
    model_name = "RFR"
    # Example of loading the dataset _________________________________________________________________
    df, ids, df_males, df_females = PD.load_dataset(path="../dataset/parkinsons_updrs.data",
                                                                  return_gender=True)

    # Normalizing/scaling  dataset
    PD.normalize_dataset(dataset=df, scaler=MinMaxScaler(), inplace=True)

    # Split dataset
    X_all = df[PD.FEATURES].values
    y_all_total = df[PD.TOTAL_UPDRS].values
    y_all_motor = df[PD.MOTOR_UPDRS].values

    # ________________________________________________________________________________________________
    # Experiment on Recursive feature elimination
    params = {'n_estimators': 500}
    model = RandomForestRegressor(**params)

    all_feature_masks, all_mae_log = PD.recursive_feature_elimination(model=model, X=X_all,
                                                                      y_total=y_all_total,
                                                                      y_motor=y_all_motor)
    print("[All] Total features: %s \n [MAE %.7f] " % (numpy.array(PD.FEATURES)[all_feature_masks['Total']],
                                                       numpy.min(all_mae_log['Total'])))
    print("[All] Motor features: %s \n [MAE %.7f] " % (numpy.array(PD.FEATURES)[all_feature_masks['Motor']],
                                                       numpy.min(all_mae_log['Motor'])))

    X_males = df_males[PD.FEATURES].values
    y_total = df_males[PD.TOTAL_UPDRS].values
    y_motor = df_males[PD.MOTOR_UPDRS].values
    male_feature_masks, male_mae_log = PD.recursive_feature_elimination(model=model, X=X_males,
                                                                        y_total=y_total,
                                                                        y_motor=y_motor)
    print("[Male] Total features: %s \n [MAE %.7f] " % (numpy.array(PD.FEATURES)[male_feature_masks['Total']],
                                                        numpy.min(male_mae_log['Total'])))
    print("[Male] Motor features: %s \n [MAE %.7f] " % (numpy.array(PD.FEATURES)[male_feature_masks['Motor']],
                                                        numpy.min(male_mae_log['Motor'])))

    X_females = df_females[PD.FEATURES].values
    y_total = df_females[PD.TOTAL_UPDRS].values
    y_motor = df_females[PD.MOTOR_UPDRS].values
    female_feature_masks, female_mae_log = PD.recursive_feature_elimination(model=model, X=X_females,
                                                                            y_total=y_total,
                                                                            y_motor=y_motor)

    print("[Female] Total features: %s \n [MAE %.7f] " % (numpy.array(PD.FEATURES)[female_feature_masks['Total']],
                                                          numpy.min(female_mae_log['Total'])))
    print("[Female] Motor features: %s \n [MAE %.7f] " % (numpy.array(PD.FEATURES)[female_feature_masks['Motor']],
                                                          numpy.min(female_mae_log['Motor'])))

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
