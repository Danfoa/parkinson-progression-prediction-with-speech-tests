import os
from copy import copy
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold


class ParkinsonDataset:
    SUBJECT_ID = "subject#"
    SUBJECT_SEC = "sex"
    SUBJECT_AGE = "age"
    TIME = "test_time"
    MOTOR_UPDRS = "motor_UPDRS"
    TOTAL_UPDRS = "total_UPDRS"
    FEATURES = ["age", "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP", "Shimmer",
                "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE",
                "DFA", "PPE"]

    @staticmethod
    def load_dataset(path="dataset/parkinsons_updrs.data", return_gender=False):
        """
        :param path: Path to dataset
        :param return_gender: True to return ids of males and females participants
        :return: 
        """
        path = os.path.join('..', path)
        df = pandas.read_csv(path, sep=',')

        # Remove instances with NAN values ____________________________________________
        data_with_missing_values = df.index[df.isna().any(axis=1)]
        if len(data_with_missing_values) > 0:
            print("Removing %.2f%% observations with missing categorical values" %
                  ((len(data_with_missing_values) / df.shape[0]) * 100))
            df = df.drop(index=data_with_missing_values)
        df.reset_index(drop=True, inplace=True)
        # _____________________________________________________________________________
        if return_gender:
            participants = numpy.unique(df["subject#"])
            print("Loaded recordings from %d participants" % len(participants))
            males = numpy.unique(df[df["sex"] == 0]["subject#"])
            females = numpy.unique(df[df["sex"] == 1]["subject#"])
            print("- %d males: %s\n- %d females %s" % (len(males), males, len(females), females))
            male_records, female_records = df[ParkinsonDataset.SUBJECT_ID] == males[0], df[ParkinsonDataset.SUBJECT_ID] == females[0]
            for id in males:
                male_records = numpy.logical_or(male_records, df[ParkinsonDataset.SUBJECT_ID] == id)

            for id in females:
                female_records = numpy.logical_or(female_records, df[ParkinsonDataset.SUBJECT_ID] == id)

            df_males = df.loc[male_records, :].copy()
            df_females = df.loc[female_records, :].copy()

            return df, participants, df_males, df_females
        else:
            return df

    @staticmethod
    def normalize_dataset(dataset, scaler=MinMaxScaler(), inplace=True):
        """
        Process dataset features by applying the provided `scaler`.
        Only the speech-recorded features are normalized/scaled
        :param dataset: Parkinson dataset dataframe
        :param scaler: Sklearn.preprocessing scaler instance
        :param inplace: True if scaling is applied directly on the dataframe provided
        :return:
            feature_normalizers - if `inplace` is True
            normalized_dataset, feature_normalizers - if `inplace` is False

            feature_normalizers: a dictionary with the scaler instance used for each feature key/name in the
            `ParkinsonDataset.FEATURES` list
        """
        if inplace:
            data = dataset
        else:
            data = dataset.copy()

        # Save encoders and scalers for each feature
        feature_normalizers = {}
        # Scale/encode all features
        for feature in ParkinsonDataset.FEATURES:
            # print(feature, end=', ')
            # Compute average and std of the data to transform
            scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))
            # Replace original data with
            data[feature] = scaled_data
            # Save scaler class in case user wants to rescale data
            # TODO: Evaluate if copy or deep copy shall be used, or only saving the parameters.
            feature_normalizers[feature] = copy(scaler)
        # print()
        if inplace:
            return feature_normalizers
        else:
            return data, feature_normalizers

    @staticmethod
    def split_dataset(dataset, train_size=0.8, test_size=0.2, subject_partitioning=False):
        """
        Partition dataset for training either randomly sampling instances of the dataset to create an
        80, 10 , 10 partition (train, test, val), or partitioning taking into account subjects ids.
        i.e. 80 % of the subjects for the training, 10% of the subjects for test, and 10% for validation.
        :param test_size:
        :param train_size:
        :param dataset: Dataset to partition
        :param subject_partitioning: True to partition data by subjects ids and not by recorded instances
        :return: Train and test as numpy nd-arrays, and target values with Total and Motor UPDRS as column vectors
            respectively
        """

        X = dataset[ParkinsonDataset.FEATURES].values
        y = dataset[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def split_dataset_indices(dataset, indices, train_size=0.8, test_size=0.2, subject_partitioning=False):
        """
        Partition dataset for training either randomly sampling instances of the dataset to create an
        80, 10 , 10 partition (train, test, val), or partitioning taking into account subjects ids.
        i.e. 80 % of the subjects for the training, 10% of the subjects for test, and 10% for validation.
        :param test_size:
        :param train_size:
        :param dataset: Dataset to partition
        :param subject_partitioning: True to partition data by subjects ids and not by recorded instances
        :return: Train and test as numpy nd-arrays, and target values with Total and Motor UPDRS as column vectors
            respectively
        """

        X = dataset[ParkinsonDataset.FEATURES].values
        y = dataset[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values

        X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, indices, train_size=train_size, test_size=test_size)

        return X_train, X_test, y_train, y_test, train_index, test_index

    @staticmethod
    def split_reduced_dataset(X, dataset, train_size=0.8, test_size=0.2, subject_partitioning=False):
        """
        Partition dataset for training either randomly sampling instances of the dataset to create an
        80, 10 , 10 partition (train, test, val), or partitioning taking into account subjects ids.
        i.e. 80 % of the subjects for the training, 10% of the subjects for test, and 10% for validation.
        :param test_size:
        :param train_size:
        :param X: Dataset instances to partition
        :param dataset: Pandas dataframe containing original dataset
        :param subject_partitioning: True to partition data by subjects ids and not by recorded instances
        :return: Train and test as numpy nd-arrays, and target values with Total and Motor UPDRS as column vectors
            respectively
        """

        y = dataset[[ParkinsonDataset.TOTAL_UPDRS, ParkinsonDataset.MOTOR_UPDRS]].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def recursive_feature_elimination(model, X, y_total, y_motor, cv=3):
        """
        Wrapper to Recursive Feature Elimination with Cross Validation for the Parkinson Telemonitoring dataset.
        :param model: Regressor model to analyse
        :param X: Training data
        :param y_total: Target Total UPDRS values
        :param y_motor: Target Motor UPDRS values
        :param cv: Number of CrossValidation folds
        :return: feature_masks, mae_log
                 - feature_masks: Dictionary holding boolean mask indicating the selected features
                 - mae_log: Dictionary holding the MAE values with different number of features.
                 Keys=["Total", "Motor"]
        """
        print("Recursive feature elimination with model: \n", model)
        feature_masks = {}
        mae_log = {}
        cv_splitter = KFold(n_splits=5, shuffle=True)
        rfecv = RFECV(estimator=model, step=1, cv=cv_splitter, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        for y_target, y_type in zip([y_total, y_motor], ['Total', 'Motor']):
            rfecv.fit(X, y_target)
            feature_masks[y_type] = rfecv.support_
            mae_log[y_type] = rfecv.grid_scores_ * -1

        return feature_masks, mae_log