from utils.dataset_loader import ParkinsonDataset as PD
import matplotlib.pyplot as plt
import pandas
import numpy
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

if __name__ == '__main__':
    print(os.getcwd())
    # Getting female and male ids
    df, ids, df_males, df_females = PD.load_dataset(path="dataset/parkinsons_updrs.data", return_gender=True)

    scaler = RobustScaler()
    _ = PD.normalize_dataset(dataset=df, scaler=MinMaxScaler(), inplace=True)
    _ = PD.normalize_dataset(dataset=df_males, scaler=MinMaxScaler(), inplace=True)
    _ = PD.normalize_dataset(dataset=df_females, scaler=MinMaxScaler(), inplace=True)

    targets = [PD.MOTOR_UPDRS, PD.TOTAL_UPDRS]

    print(df.shape, df_males.shape, df_females.shape)

    cor = df.corr()
    plt.figure(figsize=(15, 8))
    plt.subplot(311)
    plt.title("Parkinson Dataset")
    sns.heatmap(cor.loc[targets, :], annot=True, cmap=plt.cm.bone, vmin=-0.5, vmax=1)

    plt.subplot(312)
    cor = df_males.corr()
    plt.title("Parkinson Dataset (Male)")
    sns.heatmap(cor.loc[targets, :], annot=True, cmap=plt.cm.bone, vmin=-0.5, vmax=1)

    plt.subplot(313)
    cor = df_females.corr()
    plt.title("Parkinson Dataset (Females)")
    sns.heatmap(cor.loc[targets, :], annot=True, cmap=plt.cm.bone, vmin=-0.5, vmax=1)

    plt.tight_layout()
    plt.savefig("../media/features_UPDRS_corr.png")
    plt.show()

    # ______________________________________________________________________________
    model = LinearRegression()

    X_all = df[PD.FEATURES].values
    y_total = df[PD.TOTAL_UPDRS].values
    y_motor = df[PD.MOTOR_UPDRS].values
    all_feature_masks, all_mae_log = PD.recursive_feature_elimination(model=model, X=X_all,
                                                                      y_total=y_total,
                                                                      y_motor=y_motor, cv=10)
    print("[All] Total features: %s" % numpy.array(PD.FEATURES)[all_feature_masks['Total']])
    print("[All] Motor features: %s" % numpy.array(PD.FEATURES)[all_feature_masks['Motor']])

    X_males = df_males[PD.FEATURES].values
    y_total = df_males[PD.TOTAL_UPDRS].values
    y_motor = df_males[PD.MOTOR_UPDRS].values
    male_feature_masks, male_mae_log = PD.recursive_feature_elimination(model=model, X=X_males,
                                                                        y_total=y_total,
                                                                        y_motor=y_motor, cv=10)
    print("[Male] Total features: %s" % numpy.array(PD.FEATURES)[male_feature_masks['Total']])
    print("[Male] Motor features: %s" % numpy.array(PD.FEATURES)[male_feature_masks['Motor']])

    X_females = df_females[PD.FEATURES].values
    y_total = df_females[PD.TOTAL_UPDRS].values
    y_motor = df_females[PD.MOTOR_UPDRS].values
    female_feature_masks, female_mae_log = PD.recursive_feature_elimination(model=model, X=X_females,
                                                                            y_total=y_total,
                                                                            y_motor=y_motor, cv=10)

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
    title = "Recursive Feature Elimination [Linear Regression]"
    plt.title(title)
    plt.xlabel("Number of selected features")
    plt.ylabel("Cross validated average MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../media/" + title.replace(" ", "_") + ".png")
    plt.show()
