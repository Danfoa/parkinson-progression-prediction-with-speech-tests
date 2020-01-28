
# Measuring Parkinson’s disease progression

This repository studies the performance of various regressor models on the prediction of progression of the Parkinson’s disease’s total and motor UPDRS metrics. The studied models are Random Forest Regressor (RFR), Support Vector Regressor (SVR), Multi Layer Perceptron networks (MLP), Gradient Boosting (GBR) and Adaptive Neuro-fuzzy Inference System (ANFIS).
These models where tested on a variety of configurations, involving
the use of Principal Component Analysis (PCA) and Recursive Feature Elimination (RFE) for dimensionality reduction, and the use of clustering and ensembling techniques.

More information on the results and the experimental setup can be found in the attached project report.
## Dataset 
This repository uses the [Parkinson's Telemonitoring Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring) from the UCI repository. For training of the models all features from the dataset but `subject#` and `test_time` are used. 
_____________
## Instructions on how to execute our code
The scripts that can be executed in order to run our experiments can be
found under the **experiments** folder:
  - **hyperparameters\_search**: search of the hyperparameters for each
    model.
  - **models\_all\_dataset**: regression using different models and the
    original dataset.
  - **models\_projected\_dataset**: regression using different ensembled
    models and the projected datasets with Clustering + PCA.
  - **models\_recursive\_feature\_elimination**: regression using GBR and
    RFR models, with recursive feature elimination (RFE).
  - **reduction**: inside there is the script that projects (reduces) the
    dataset using PCA and the different clustering algorithms.

These scripts print an output with their results and save them in the
folder **results**. Under **utils** you can find scripts that, with the
results obtained by the experiments, generate plots comparing the
different algorithms. The plots are saved in **media**. The project was developed with Python 3 (we recommend Python >= 3.6).