from utils.dataset_loader import ParkinsonDataset as PD
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os
if __name__ == '__main__':
    print(os.getcwd())
    # Example of loading the dataset
    data_frame = PD.load_dataset(path="dataset/parkinsons_updrs.data",
                                 return_gender=False)
    # Getting female and male ids
    df, ids, df_males, df_females = PD.load_dataset(path="dataset/parkinsons_updrs.data", return_gender=True)

    PD.normalize_dataset(dataset=df, scaler=MinMaxScaler(), inplace=True)
    PD.normalize_dataset(dataset=df_females, scaler=MinMaxScaler(), inplace=True)
    PD.normalize_dataset(dataset=df_males, scaler=MinMaxScaler(), inplace=True)
