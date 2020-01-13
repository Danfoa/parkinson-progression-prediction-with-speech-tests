import matplotlib.pyplot as plt
import pandas
import numpy as np

paths = ["../results/outputs/GBR/MAE-diff-components.csv", "../results/outputs/RFR/MAE-diff-components.csv",
         "../results/outputs/SVR/MAE-diff-components.csv"]


def plot_diff_pca_components(paths):
    motor = []
    for path in paths:
        df = pandas.read_csv(path, usecols=[0, 1, 3])
        results = [df["Unnamed: 0"].tolist(), df["Total-Test"].tolist(), df["Motor-Test"].tolist()]
        plt.plot(results[0], np.absolute(results[1]), marker=".")
        plt.plot(results[0], np.absolute(results[2]), marker=".")

    plt.legend(['GBR Total_UPDRS', 'GBR Motor_UPDRS', 'RFR Total_UPDRS', 'RFR Motor_UPDRS', 'SVR Total_UPDRS',
                'SVR Motor_UPDRS'], loc='upper right')
    plt.xlabel("Number of components")
    plt.ylabel("MAE")
    plt.savefig("../media/MAE-diff-components_models_comparison.png")
    plt.show()
    return


plot_diff_pca_components(paths)
