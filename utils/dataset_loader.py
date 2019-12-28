import numpy
import pandas
from copy import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


class ParkinsonDataset:
    SUBJECT_ID = "subject#"
    SUBJECT_SEC = "sex"
    SUBJECT_AGE = "age"
    TIME = "test_time"
    MOTOR_UPDRS = "motor_UPDRS"
    TOTAL_UPDRS = "total_UPDRS"
    FEATURES = ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP", "Shimmer", "Shimmer(dB)",
                "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]

    @staticmethod
    def load_dataset(path="dataset/parkinsons_updrs.data", return_gender=False):
        """
        :param path: Path to dataset
        :param return_gender: True to return ids of males and females participants
        :return: 
        """
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
            return df, participants, males, females
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
            # Compute average and std of the data to transform
            scaler.fit(data[feature].values.reshape(-1, 1))
            # Replace original data with
            data[feature] = scaler.transform(data[feature].values.reshape(-1, 1))
            # Save scaler class in case user wants to rescale data
            # TODO: Evaluate if copy or deep copy shall be used, or only saving the parameters.
            feature_normalizers[feature] = copy(scaler)
        if inplace:
            return feature_normalizers
        else:
            return data, feature_normalizers

    @staticmethod
    def split_dataset(dataset, subject_partitioning=False):
        """
        Partition dataset for training either randomly sampling instances of the dataset to create an
        80, 10 , 10 partition (train, test, val), or partitioning taking into account subjects ids.
        i.e. 80 % of the subjects for the training, 10% of the subjects for test, and 10% for validation.
        :param dataset: Dataset to partition
        :param subject_partitioning: True to partition data by subjects ids and not by recorded instances
        :return: Train, test, and Validation as numpy nd-arrays, and target values for Total and Motor UPDRS
        """
        # TODO: Implement
        X_train, X_test, X_val = None, None, None
        y_train_total, y_test_total, y_val_total = None, None, None
        y_train_motor, y_test_motor, y_val_motor = None, None, None
        y_total_UPDRS = (y_train_total, y_test_total, y_val_total)
        y_motor_UPDRS = (y_train_motor, y_test_motor, y_val_motor)
        return X_train, X_test, X_val, y_total_UPDRS, y_motor_UPDRS
