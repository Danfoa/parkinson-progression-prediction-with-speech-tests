import seaborn as sb
import matplotlib.pyplot as plt
from statistics import mean
import pandas
import csv


def convertList(list):
    a = list.replace("[", " ")
    a = a.replace("]", " ")
    a = a.split(",")
    return a


def readValues(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        total = []
        motor = []
        for row in reader:
            total_mean = convertList(row[1])
            motor_mean = convertList(row[2])

            total_mean = mean([float(i) for i in total_mean])
            motor_mean = mean([float(i) for i in motor_mean])

            total.append(total_mean)
            motor.append(motor_mean)
    return [total, motor]


def change_width(ax, new_value):
    print(len(ax.patches))
    for i in range(len(ax.patches)):

        current_width = ax.patches[i].get_height()
        diff = current_width - new_value

        # we change the bar width
        ax.patches[i].set_height(new_value)

        # we recenter the bar
        if i < 3:
            ax.patches[i].set_y(ax.patches[i].get_y() + diff)



def makePlot(total, motor, model):
    df = pandas.DataFrame(
        {'Clusters': ['fuzzy', 'som', 'em'], 'Total_UPDRS': total,
         'Motor_UPDRS': motor})
    fig, ax1 = plt.subplots(figsize=(10, 2.5))
    tidy = df.melt(id_vars='Clusters').rename(columns=str.title)

    sn = sb.barplot(x='Value', y='Clusters', hue='Variable', data=tidy, ax=ax1, palette='bone')

    sn.axes.set_title(model, fontsize=22)
    sn.set_xlabel("MAE", fontsize=18)
    sn.set_ylabel("", fontsize=18)
    sb.despine(fig)

    change_width(sn, 0.3)
    ax1.get_legend().remove()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("../media/"+ model +"_cluster_comparison.png")
    plt.show()

    return


ANFIS_values = readValues("../results/outputs/ANFIS/MAE-clustering+regression_results.csv")
GBR_values = readValues("../results/outputs/GBR/MAE-clustering+regression_results.csv")
MLP_values = readValues("../results/outputs/MLP/MAE-clustering+regression_results.csv")
RFR_values = readValues("../results/outputs/RFR/MAE-clustering+regression_results.csv")
SVR_values = readValues("../results/outputs/SVR/MAE-clustering+regression_results.csv")

makePlot(ANFIS_values[0], ANFIS_values[1], "ANFIS")
makePlot(GBR_values[0], GBR_values[1], "GBR")
makePlot(MLP_values[0], MLP_values[1], "MLP")
makePlot(RFR_values[0], RFR_values[1], "RFR")
makePlot(SVR_values[0], SVR_values[1], "SVR")


