
# Measuring Parkinson’s disease progression (CI - MAI)
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
different algorithms. The plots are saved in **media**. To execute all
our code it is needed to install Python 3 (we recommend Python >= 3.6, at is the one we have used) and some external libraries:

  - numpy
  - sklearn
  - skfuzzy
  - fuzzy-c-means
  - pandas
  - tensorflow
  - seaborn
  - matplotlib