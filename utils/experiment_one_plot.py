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


values = [readValues("../results/outputs/GBR/MAE-final-GBR-KFold.csv"),
          readValues("../results/outputs/MLP/MAE-final-MLP-KFold.csv"),
          readValues("../results/outputs/RFR/MAE-final-RFR-KFold.csv"),
          readValues("../results/outputs/SVR/MAE-final-SVR-KFold.csv")]
values = list(map(list, zip(*values)))

df = pandas.DataFrame({
    'Models': ["GBR", "MLP", "RFR", "SVR"],
    'Total_UPDRS': values[0],
    'Motor_UPDRS': values[1]
})
fig, ax1 = plt.subplots(figsize=(10, 10))
tidy = df.melt(id_vars='Models').rename(columns=str.title)
seaborn.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1)
seaborn.despine(fig)
plt.savefig("../media/Ensembling_experiment_all_models.png")
plt.show()

df = pandas.DataFrame({
    'Models': ["GBR", "MLP", "RFR", "SVR"],
    'Total_UPDRS_male': values[2],
    'Motor_UPDRS_male': values[3]
})
fig, ax1 = plt.subplots(figsize=(10, 10))
tidy = df.melt(id_vars='Models').rename(columns=str.title)
seaborn.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1)

seaborn.despine(fig)
plt.savefig("../media/Ensembling_experiment_all_models_male.png")
plt.show()

df = pandas.DataFrame({
    'Models': ["GBR", "MLP", "RFR", "SVR"],
    'Total_UPDRS_female': values[4],
    'Motor_UPDRS_female': values[5]
})
fig, ax1 = plt.subplots(figsize=(10, 10))
tidy = df.melt(id_vars='Models').rename(columns=str.title)
seaborn.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1)
seaborn.despine(fig)
plt.savefig("../media/Ensembling_experiment_all_models_female.png")
plt.show()