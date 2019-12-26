import numpy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

# Custom scripts
import utils



if __name__ == '__main__':

    df, ids, males, females = utils.load_dataset()

    plt.figure(figsize=(15, 15))
    sns.pairplot(df[utils.features], kind="reg", palette=plt.bone())
    plt.show()


