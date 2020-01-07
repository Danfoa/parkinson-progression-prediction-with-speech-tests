import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler

from utils.dataset_loader import ParkinsonDataset
from sklearn.decomposition import PCA


def get_reduced_dataset(cluster_data, all_data):
    pca = PCA(.95)
    pca.fit_transform(cluster_data)
    return pca.transform(all_data)


if __name__ == '__main__':
    load_path = "../../results/clustering/"
    save_path = "../../results/reduction/"

    clustering_algorithms = ["fuzzy_c_means", "som", "em"]
    clusters = [[4, 5], [4, 9], [4, 12]]

    # loading the dataset
    df = ParkinsonDataset.load_dataset(path="../dataset/parkinsons_updrs.data",
                                       return_gender=False)

    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=MinMaxScaler(),
                                                             inplace=True)

    X = df[ParkinsonDataset.FEATURES].values

    for i in range(len(clustering_algorithms)):
        algorithm = clustering_algorithms[i]
        for c in clusters[i]:
            labels = numpy.load(load_path + algorithm + '/C=%d-labels.npy' % c)
            for k in range(c):
                reduced_dataset = get_reduced_dataset(X[numpy.where(labels == k)], X)
                numpy.save(save_path + algorithm + '/C=%d-K=%d-reduced-dataset.npy' % (c, k), reduced_dataset)