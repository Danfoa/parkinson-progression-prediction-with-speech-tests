import matplotlib.pyplot as plt
import pandas
import seaborn as sb

sb.set(style="whitegrid")


def makePlot(total, motor):
    df = pandas.DataFrame(
        {'Models': ['EM+Anfis', 'fuzzy+GBR', 'som+MLP', 'fuzzy+RFR', 'som+SVR'],
         'Total_UPDRS': total,
         'Motor_UPDRS': motor})
    fig, ax1 = plt.subplots(figsize=(15, 11))
    tidy = df.melt(id_vars='Models').rename(columns=str.title)

    sn = sb.barplot(x='Models', y='Value', hue='Variable', data=tidy, ax=ax1, palette='bone')

    sn.axes.set_title("Clustering+Model", fontsize=22)
    sn.set_ylabel("MAE", fontsize=18)
    sn.set_xlabel("Models", fontsize=18)
    sb.despine(fig)

    plt.setp(ax1.get_legend().get_texts(), fontsize='17')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("../media/model_cluster_comparison.png")
    plt.show()

    return


total = [7.903273018514029, 4.110263370760881, 2.8862764088386768, 4.058391171758299, 5.10827984495852]
motor = [6.12038805854495, 3.266110069906717, 2.2763016439621944, 3.18454233986042, 3.9510130319342722]

makePlot(total, motor)
