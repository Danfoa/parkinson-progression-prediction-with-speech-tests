from utils.dataset_loader import ParkinsonDataset as PD
import matplotlib.pyplot as plt
import pandas
import numpy
import seaborn as sns
import os


if __name__ == '__main__':

    print(os.getcwd())
    # Getting female and male ids
    df, ids, df_males, df_females = PD.load_dataset(path="dataset/parkinsons_updrs.data", return_gender=True)

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

