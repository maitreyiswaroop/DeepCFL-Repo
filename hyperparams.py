import torch
import numpy as np
torch_seed = 17#30291709#300970#1318995137987573659
torch.manual_seed(torch_seed)
np_seed = 17#30291709#171000#307102485//(2**32)
np.random.seed(np_seed)

# colors for plots
colors = ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd"]

device = torch.device("mps")
l1_sum_dim = 0

# results_dir = f"DeepCFL Expts/Many-One/Seed_{np_seed}"
results_dir = f"results"

separate_optimizers = False
alt_opt_update = False
alt_opt_update_freq = 2
scheduler = False
lr = 1e-3
lr_fxy = 1e-3
epochs = 20
bs = 256

# Data dimensions
dim_x_h = 10    # dimension of macro cause
dim_y_h = 10    # dimension of macro effect

classes_x = 10  # specific to MNIST experiment, number of classes in X
classes_y = 10  # specific to MNIST experiment, number of classes in Y

# VAE parameters, both Kannada MNIST and English MNIST datasets have 28x28 images
# Using the same architecture for both X and Y
input_shape = [1,28, 28]  # (channels, height, width)
conv_filters = [32,64,128,256]  # number of filters in each layer
conv_kernels = [3, 3, 3, 3]  # kernel sizes in each layer
conv_strides = [2, 2, 2, 2]  # strides for each layer
padding = [1, 1, 1, 1]  # padding sizes in each layer
fc = []
x_encoder_specs =   (conv_filters,  conv_kernels,   conv_strides,   padding,fc)
fc=[1024]
conv_filters = [128, 64, 32, 1]  # number of filters in each layer
conv_kernels = [3, 3, 3, 3]  # kernel sizes in each layer
conv_strides = [2, 2, 2, 2]  # strides in each layer
padding = [1, 1, 1, 1]  # padding sizes in each layer
output_padding = [1, 0, 1, 1]  # padding sizes in each layer
reshape=[256, 2, 2]
x_decoder_specs = (conv_filters,conv_kernels,conv_strides,padding,output_padding,reshape,fc)


# Model parameters
fxy_type = 'Linear' 

with_softmax = True

# Scheduler for lambda hyperparameter
scheduler = 'Sigmoid'
# scheduler_type = 'Sigmoid'
lambda_range = [1.0, 10.0]

# Beta values for the beta-VAEs
beta_x = 1.0
beta_y = 2.0

# Sparsity penalty
lambda_2 = 0.0