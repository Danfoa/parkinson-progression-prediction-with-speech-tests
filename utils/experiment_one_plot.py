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
        total = []
        motor = []
        for row in reader:
            total_mean = convertList(row[1])
            motor_mean = convertList(row[2])

            total.append(mean([float(i) for i in total_mean]))
            motor.append(mean([float(i) for i in motor_mean]))

    return total, motor


total_GBR, motor_GBR = readValues("../results/outputs/GBR/MAE-final-GBR-KFold.csv")
total_MLP, motor_MLP = readValues("../results/outputs/MLP/MAE-final-MLP-KFold.csv")
total_RFR, motor_RFR = readValues("../results/outputs/RFR/MAE-final-RFR-KFold.csv")
total_SVR, motor_SVR = readValues("../results/outputs/SVR/MAE-final-SVR-KFold.csv")


df = pandas.DataFrame({
    'Models': ["GBR", "MLP", "RFR", "SVR"],
    'Total_UPDRS': [total_GBR[0], total_MLP[0], total_RFR[0], total_SVR[0]],
    'Total_UPDRS_Males': [total_GBR[1], total_MLP[1], total_RFR[1], total_SVR[1]],
    'Total_UPDRS_Females': [total_GBR[2], total_MLP[2], total_RFR[2], total_SVR[2]],
})
fig, ax1 = plt.subplots(figsize=(15, 11))
tidy = df.melt(id_vars='Models').rename(columns=str.title)

sn = seaborn.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1, palette ='bone')
sn.axes.set_title("Total_UPDRS Model comparison",fontsize=22)
sn.set_xlabel("Models",fontsize=18)
sn.set_ylabel("MAE",fontsize=18)
seaborn.despine(fig)

plt.setp(ax1.get_legend().get_texts(), fontsize='17')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig("../media/total_UPDRS_Model_comparison.png")
plt.show()

df = pandas.DataFrame({
    'Models': ["GBR", "MLP", "RFR", "SVR"],
    'Motor_UP': [motor_GBR[0], motor_MLP[0], motor_RFR[0], motor_SVR[0]],
    'Motor_UPDRS_Males': [motor_GBR[1], motor_MLP[1], motor_RFR[1], motor_SVR[1]],
    'Motor_UPDRS_Females': [motor_GBR[2], motor_MLP[2], motor_RFR[2], motor_SVR[2]],

})

fig, ax1 = plt.subplots(figsize=(15, 11))
tidy = df.melt(id_vars='Models').rename(columns=str.title)

sn = seaborn.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1, palette ='bone')
sn.axes.set_title("Motor_UPDRS Model comparison",fontsize=22)
sn.set_xlabel("Models",fontsize=18)
sn.set_ylabel("MAE",fontsize=18)
seaborn.despine(fig)

plt.setp(ax1.get_legend().get_texts(), fontsize='17')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig("../media/motor_UPDRS_Model_comparison.png")
plt.show()
