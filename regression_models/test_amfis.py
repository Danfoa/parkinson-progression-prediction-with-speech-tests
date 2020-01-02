import os
import numpy as np
from regression_models import amfis_model

if __name__ == '__main__':
    path = os.path.join("../", "reduced_datasets/datasets_" + str(0)+ '.npy')
    data = np.load(path)
    path = os.path.join("../", "reduced_datasets/datasets_" + 'y' + '.npy')
    y = np.load(path)
    model = amfis_model.AMFIS(data, y)
    model.fit()
