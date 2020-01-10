"""
Each regression model is going to be tested with different pre-processing and reduction techniques and hyper-parameters
the below script will contain all the model independent code.
"""

import pandas
import numpy

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Custom imports
from utils.dataset_loader import ParkinsonDataset
from utils.experiment_one_plot import *

if __name__ == '__main__':
    # Example of loading the dataset
    df = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                       return_gender=False)
    # Normalizing/scaling  dataset
    feature_normalizers = ParkinsonDataset.normalize_dataset(dataset=df,
                                                             scaler=MinMaxScaler(),
                                                             inplace=True)
    # Split dataset
    X_train, X_test, y_train, y_test = ParkinsonDataset.split_dataset(dataset=df,
                                                                      subject_partitioning=False)
    # Get TOTAL UPDRS targets
    y_train_total, y_test_total = y_train[:, 0], y_test[:, 0]
    # Get MOTOR UPDRS targets
    y_train_motor, y_test_motor = y_train[:, 1], y_test[:, 1]

    # Dimensionality reduction techniques ____________________________________________________________
    # TODO
    # Training TOTAL UPDRS____________________________________________________________________________
    # TODO
    # Training MOTOR UPDRS____________________________________________________________________________
    # TODO
    # Saving results for comparison __________________________________________________________________
    # TODO
