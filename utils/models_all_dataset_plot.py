import seaborn as sn
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
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

            aux_total = [float(i) for i in total_mean]
            aux_motor = [float(i) for i in motor_mean]

            st_total = np.std(aux_total)
            st_motor = np.std(aux_motor)

            total.append(mean(aux_total))
            motor.append(mean(aux_motor))

    return total, motor, st_total, st_motor


total_GBR, motor_GBR, std_total_GBR, std_motor_GBR = readValues("../results/outputs/GBR/MAE-final-GBR-KFold.csv")
total_MLP, motor_MLP, std_total_MLP, std_motor_MLP = readValues("../results/outputs/MLP/MAE-final-MLP-KFold.csv")
total_RFR, motor_RFR, std_total_RFR, std_motor_RFR = readValues("../results/outputs/RFR/MAE-final-RFR-KFold.csv")
total_SVR, motor_SVR, std_total_SVR, std_motor_SVR = readValues("../results/outputs/SVR/MAE-final-SVR-KFold.csv")

#std = [std_total_GBR, std_motor_GBR, std_total_MLP, std_motor_MLP, std_total_RFR, std_motor_RFR, std_total_SVR, std_motor_SVR]

sn.set(style="whitegrid")
df = pandas.DataFrame({
    'Models': ["GBR", "MLP", "RFR", "SVR"],
    'Total_UPDRS': [total_GBR[0], total_MLP[0], total_RFR[0], total_SVR[0]],
    'Total_UPDRS_Males': [total_GBR[1], total_MLP[1], total_RFR[1], total_SVR[1]],
    'Total_UPDRS_Females': [total_GBR[2], total_MLP[2], total_RFR[2], total_SVR[2]],
})
fig, ax1 = plt.subplots(figsize=(15, 11))
tidy = df.melt(id_vars='Models').rename(columns=str.title)

figure = sn.barplot(x='Models', y='Value', hue='Variable', ci="sd",  data=tidy, ax=ax1, palette='bone')
figure.axes.set_title("Total_UPDRS Model comparison", fontsize=22)
figure.set_xlabel("Models", fontsize=18)
figure.set_ylabel("MAE", fontsize=18)
sn.despine(fig)

plt.setp(ax1.get_legend().get_texts(), fontsize='17')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig("../media/total_UPDRS_Model_comparison.png")
plt.show()

df = pandas.DataFrame({
    'Models': ["GBR", "MLP", "RFR", "SVR"],
    'Motor_UPDRS': [motor_GBR[0], motor_MLP[0], motor_RFR[0], motor_SVR[0]],
    'Motor_UPDRS_Males': [motor_GBR[1], motor_MLP[1], motor_RFR[1], motor_SVR[1]],
    'Motor_UPDRS_Females': [motor_GBR[2], motor_MLP[2], motor_RFR[2], motor_SVR[2]],

})

fig, ax1 = plt.subplots(figsize=(15, 11))
tidy = df.melt(id_vars='Models').rename(columns=str.title)
sn.set(style="whitegrid")
figure = sn.barplot(x='Models', y='Value', hue='Variable', ci="sd",  data=tidy, ax=ax1, palette='bone')
figure.axes.set_title("Motor_UPDRS Model comparison", fontsize=22)
figure.set_xlabel("Models", fontsize=18)
figure.set_ylabel("MAE", fontsize=18)
sn.despine(fig)

plt.setp(ax1.get_legend().get_texts(), fontsize='17')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig("../media/motor_UPDRS_Model_comparison.png")
plt.show()




