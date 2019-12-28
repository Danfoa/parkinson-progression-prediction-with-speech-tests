import numpy
import pandas

class DatasetLoader:

    # subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP,Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA,NHR,HNR,RPDE,DFA,PPE

    sub_features = ["subject#", "age", "sex"]
    features = ["age", "test_time", "motor_UPDRS", "total_UPDRS", "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5",
                "Jitter:DDP", "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
                "NHR", "HNR", "RPDE", "DFA", "PPE"]

    # def load_temporal_dataset(path="Dataset/parkinsons_updrs.data"):
    #     df = pandas.read_csv(path, sep=',')
    #
    #     data_with_missing_values = df.index[df.isna().any(axis=1)]
    #     if len(data_with_missing_values) > 0:
    #         print("Removing %.2f%% observations with missing categorical values" %
    #               ((len(data_with_missing_values) / df.shape[0]) * 100))
    #         df = df.drop(index=data_with_missing_values)
    #     df.reset_index(drop=True, inplace=True)
    #
    #     # print(df.head())
    #     participants = numpy.unique(df["subject#"])
    #     print("Loaded recordings from %d participants" % len(participants))
    #     males = numpy.unique(df[df["sex"] == 0]["subject#"])
    #     females = numpy.unique(df[df["sex"] == 1]["subject#"])
    #     print("- %d males: %s\n- %d females %s" % (len(males), males, len(females), females))
    #     return df, participants, males, females

    def load_dataset(path="Dataset/parkinsons_updrs.data"):
        df = pandas.read_csv(path, sep=',')

        data_with_missing_values = df.index[df.isna().any(axis=1)]
        if len(data_with_missing_values) > 0:
            print("Removing %.2f%% observations with missing categorical values" %
                  ((len(data_with_missing_values) / df.shape[0]) * 100))
            df = df.drop(index=data_with_missing_values)
        df.reset_index(drop=True, inplace=True)
        return df