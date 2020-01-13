import numpy
import pandas
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Custom imports
from utils.dataset_loader import ParkinsonDataset
from utils.models_all_dataset_plot import *
# from clustering_models.fuzzy_c_means import FuzzyCMeans
from fcmeans import FCM


def get_clusters_variances(dataset, output_labels, feature_names):
    """
    Function to obtain feature variance (of observations within cluster) and cluster size of the K clusters found.
    :param dataset: Pandas `DataFrame` containing all the features of observations
    :param output_labels: Clustering algorithm output labels for all the observations in `dataset`
    :return: A Pandas dataset with the following information:
                                  cluster1   cluster2 ...  clusterK
                std(Feature 1)       -          -     ...     -
                std(Feature 1)       -          -     ...     -
                ...                 ...        ...    ...    ...
                std(Feature N)       -          -     ...     -
                cluster size         -          -     ...     -

    """

    labels = numpy.unique(output_labels)
    indexes = ["std(%s)" % feature for feature in feature_names]
    indexes = numpy.append(indexes, ['cluster_size'])
    clusters_outcome = pandas.DataFrame(columns=labels,
                                        index=indexes)
    for label in labels:
        # Use output labels to get the observations on each computed cluster
        cluster_instances = dataset[output_labels == label, :]
        cluster_data = numpy.std(cluster_instances, axis=0)
        clusters_outcome[label] = numpy.append(cluster_data, cluster_instances.shape[0])

    return clusters_outcome


if __name__ == '__main__':

    # Example of loading the dataset
    df = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                       return_gender=False)

    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=MinMaxScaler(),
                                                             inplace=True)
    print(df.head())
    # Get dataset features to clusterize
    X = df[ParkinsonDataset.FEATURES].values
    # print(X[:5, :])

    # Experiment on number of clusters
    num_trials = 10
    average_metrics_df = pandas.DataFrame()

    clusters = numpy.arange(2, 20)
    calinski, daveis = [], []

    for c in clusters:
        c_entropy, daveis_score, calinski_score = 0, 0, 0
        for trial in range(num_trials):
            fz = FCM(n_clusters=c, m=2.0, error=1.e-9)
            fz.fit(X)
            labels = fz.u.argmax(axis=1)

            # c_entropy += fz.compute_partition_entropy()
            calinski_score += calinski_harabasz_score(X=X, labels=labels)
            daveis_score += davies_bouldin_score(X=X, labels=labels)
        calinski += [calinski_score/num_trials]
        daveis += [daveis_score/num_trials]

    fig, ax1 = plt.subplots()
    ax1.plot(clusters, daveis, label='Davies&Bouldin', color='k')
    ax1.set_ylabel('Davies&Bouldin range ')
    ax1.tick_params('y')
    ax2 = ax1.twinx()
    ax2.plot(clusters, calinski, label='Calinski')
    # ax2.plot(clusters, entropy, label='Partition Entropy', color='c')
    ax2.set_ylabel('Calinski range')
    ax2.tick_params('y')

    fig.tight_layout()
    plt.grid()

    ax2.legend(loc='center right')
    ax1.legend(loc='upper right')
    ax1.set_xlabel("Number of clusters")
    plt.title("Fuzzy c-means average clustering performance")
    plt.show()

    # Cluster and save results with optimal number of clusters
    path = "../../results/clustering/fuzzy_c_means/"
    for c in [8, 10, 11, 13, 3]:
        print("Saving clustering outcome with C=%d" % c)
        fz = FCM(n_clusters=c, m=2.0, error=1.e-9)
        fz.fit(X)

        centroids = fz.centers
        labels = fz.u.argmax(axis=1)
        membership = fz.u

        print(numpy.unique(labels))
        clusters_info = get_clusters_variances(X, output_labels=labels, feature_names=ParkinsonDataset.FEATURES)
        print(clusters_info)
        numpy.save(path + "C=%d-labels" % c, labels)
        numpy.save(path + "C=%d-probabilities" % c, membership)
        numpy.save(path + "C=%d-centroids" % c, centroids)




