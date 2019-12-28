from utils.dataset_loader import ParkinsonDataset


if __name__ == '__main__':

    # Example of loading the dataset
    data_frame = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                               return_gender=False)
    # Getting female and male ids
    data_frame, ids, males, females = ParkinsonDataset.load_dataset(path="dataset/parkinsons_updrs.data",
                                                                    return_gender=True)

    print(data_frame.head())
