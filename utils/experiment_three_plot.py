"""
Functions regarding plotting ... remeber to develop scripts to automatically save plots in the `media` folder.

"""

import seaborn
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
        result = []
        for row in reader:
            total_mean = convertList(row[1])
            motor_mean = convertList(row[2])

            result.append(mean([float(i) for i in total_mean]))
            result.append(mean([float(i) for i in motor_mean]))

    return result


values = [readValues("../results/outputs/ANFIS/MAE-clustering+regression_results.csv"),
          readValues("../results/outputs/GBR/MAE-clustering+regression_results.csv"),
    #      readValues("../results/outputs/MLP/MAE-clustering+regression_results.csv"),
          readValues("../results/outputs/RFR/MAE-clustering+regression_results.csv"),
          readValues("../results/outputs/SVR/MAE-clustering+regression_results.csv")]
values = list(map(list, zip(*values)))

df = pandas.DataFrame({
    'Models': ["ANFIS","GBR", "RFR", "SVR"],
    'Total_UPDRS': values[0],
    'Motor_UPDRS': values[1]
})
fig, ax1 = plt.subplots(figsize=(10, 10))
tidy = df.melt(id_vars='Models').rename(columns=str.title)
seaborn.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1)
seaborn.despine(fig)
plt.title("Fuzzy-c-means clustering results")
plt.savefig("../media/clustering_fuzzy_c_means_all_models.png")
plt.show()

df = pandas.DataFrame({
    'Models': ["ANFIS","GBR", "RFR", "SVR"],
    'Total_UPDRS': values[2],
    'Motor_UPDRS': values[3]
})
fig, ax1 = plt.subplots(figsize=(10, 10))
tidy = df.melt(id_vars='Models').rename(columns=str.title)
seaborn.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1)

seaborn.despine(fig)
plt.title("SOM clustering results")
plt.savefig("../media/clustering_som_all_models.png")
plt.show()

df = pandas.DataFrame({
    'Models': ["ANFIS", "GBR", "RFR", "SVR"],
    'Total_UPDRS': values[4],
    'Motor_UPDRS': values[5]
})
fig, ax1 = plt.subplots(figsize=(10, 10))
tidy = df.melt(id_vars='Models').rename(columns=str.title)
seaborn.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1)
seaborn.despine(fig)
plt.title("EM clustering results")
plt.savefig("../media/clustering_em_all_models.png")
plt.show()