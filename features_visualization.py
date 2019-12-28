import numpy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

# Custom scripts
from utils import DatasetLoader



if __name__ == '__main__':

    df, ids, males, females = DatasetLoader.load_dataset()

    plt.figure(figsize=(15, 15))
    sns.pairplot(df[DatasetLoader.features], kind="reg", palette=plt.bone())
    plt.show()


