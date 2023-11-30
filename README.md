# DeepCFL-Repo

This repository contains the implementation of the DeepCFL model proposed in **Learning macro variables using auto encoders**, by *Dhanya Sridhar, Eric Elmoznino, Maitreyi Swaroop*, (poster at the NeurIPS 2023 Workshop on Causal Representation Learning).

## Libraries Used
Our python code uses the pytorch library for implementing the proposed model. 
We have set the device to "mps" in the hyperparams.py file for speedup, if mps is not enabled, you may set the device to "cpu".

## Folders and files
1. **dataset** (folder) contains the English MNIST dataset (```x_data.pt```), the paired Kannada MNIST dataset (```y_data.pt```) and the data labels (```class_data.pt```)
2. **results** (folder) will contain the results of all the experiments. Each experiment is named ```run_{experiment number}```. We have included the results of a sample run in the folder ```run_1```. This contains the following
    1. **nets** (folder): the model (```model.pt```), as well as auxiliary classifiers used to evaluate the learned representations.
    2. **reconstructions** (folder): contains the reconstructions of the digits from the latent variables ```xh, yh, yh_prime``` (kindly refer to paper for further clarification). For reconstructions of all digits, the first 10 rows contain the original digits and the last 10 rows contain their reconstructions
    3. **tsne** (folder): the tsne plots for the learned latents (macro variables)
    4. **tsne** (folder): the violin plots for the learned latents (macro variables)
    5. **curves.png**: the running losses over training
    6. **f_xy.txt**: the model overview
    7. **hyperparameters.txt**: summary of the hyperparameters for the current run
    8. **scheduler.png**: the plot of the values of lambda over the epochs when using a weight scheduler
    9. **silhouette_scores.txt**: the values of the silhouette scores for the macro variables ```xh, yh``` and the predicted macro variable ```yh_prime```
3. **data_visualisers.py**: functions to plot data, reconstructions, violin plots etc.
4. **data.py**: functions obtain the dataset for the one-to-one and many-to-one experiments
5. **experiment1.py**: python file which executes the experiment for the one-to-one case
6. **hyperparams.py**: python file which contains all the hyperparameters, model specifications and device specifications for the experiments. 
7. **models.py**: python file for the models used in our experiments
8. **objectives.py**: python file for loss functions
9. **schedulers.py**: python file for weight schedulers (used for the weight of the reconstruction term in the loss function)

## Clarifications
For any clarifications regarding the code, feel free to reach out at maitreyiswaroop@gmail.com

