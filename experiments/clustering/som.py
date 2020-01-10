from sklearn.preprocessing import MinMaxScaler

from clustering_models.em import ExpectationMaximization
from clustering_models.som import SelfOrganizingMap
from utils.dataset_loader import ParkinsonDataset

SOM_CLUSTERS = 9  # According to Nilashi2019 paper


def __train_som_model(data, num_clusters=None):
    model = SelfOrganizingMap(data, num_clusters)
    assignations = model.clusterize()
    return model, assignations


if __name__ == '__main__':
    # Example of loading the dataset
    df = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                       return_gender=False)

    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=MinMaxScaler(),
                                                             inplace=True)
    # Get dataset features to clusterize
    X = df[ParkinsonDataset.FEATURES].values

    __train_som_model(X, SOM_CLUSTERS)
