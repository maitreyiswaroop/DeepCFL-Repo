import torch
import numpy as np
from matplotlib import pyplot as plt
import colorsys

def darken_color(color, percentage):
    """
        Function to darken colour
    """
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(0, min(1, l - percentage / 100))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

# function to convert flattened image to image
def convert_to_image(image):
    """
        convert flattened image to image
    """
    if torch.is_tensor(image):
        if len(image.shape)==3:
            image=image.squeeze()
            return image
    dims = int(np.sqrt(image.shape[0]))
    image = image.view(dims, dims)
    return image

# function to plot the images
def plot_images(images, labels, save=False,fig_size=None):
    """
        Input:
            images is a list of flattened images
            labels is a list of labels
    """
    if fig_size is not None and fig_size[1]>1:
        fig, axs = plt.subplots(fig_size[1], fig_size[0], figsize=(fig_size[0], fig_size[1]))
        for j in range(fig_size[1]):
            for i in range(fig_size[0]):
                axs[j, i].imshow(convert_to_image(images[i+fig_size[0]*j]))
                axs[j, i].axis('off')
                axs[j, i].set_title(labels[i+fig_size[0]*j].numpy())
    else:
        fig = plt.figure(figsize=fig_size)
        for i in range(len(images)):
            plt.subplot(fig_size[1], len(images), i+1)
            plt.imshow(convert_to_image(images[i]))
            plt.title(labels[i].numpy())
            plt.axis('off')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    else:
        plt.show()
    plt.clf()

def violin_plts(latent_reps, labels, save_path=None, num_digits=10, plt_title=None):
    """
        Function to plot violin plots of the latent representations
        to see the activation of each dimension for different classes
    """
    with torch.no_grad():
        # dimensions of latent vectors
        num_components = latent_reps.shape[1]
        # max value
        max_val = torch.max(latent_reps)
        # min value
        min_val = torch.min(latent_reps)
        # print(f"labels = {labels}")
        # print(f"latent_reps.shape = {latent_reps.shape}")

        # storing the latent vectors for each digit
        latent_by_digit = [[] for i in range(num_digits)]
        for i in range(len(latent_reps)):
            label = int(labels[i])
            latent_by_digit[label].append(latent_reps[i]) 
        latent_by_digit=[torch.stack(inner_list) for inner_list in latent_by_digit]

        fig,axs = plt.subplots(num_components, 1, figsize=(num_digits,5*num_components))
        colors = ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd"]
        colors = [darken_color(color, 40) for color in colors]
        if plt_title is not None:
            fig.suptitle(plt_title)
        for i in range(num_components):
            # print(f"debug in dataset_utils latent_by_digit[0].shape{latent_by_digit.shape}")
            # latent_by_digit is a list of size 10 of tensors
            # the jth tensor is of size (num_samples_of_digit_j, latent_dim)
            # latent_by_digit[label][:,i] gives the ith component of the latent representations of the digit label
            # data is thus a list of 10 tensors, data[k] is the ith component of the latent representations of the digit k
            data = [latent_by_digit[label][:,i] for label in range(num_digits)] 

            axs[i].violinplot(data,showmeans=True,showmedians=True)
            for pc, color in zip(axs[i].collections, colors):
                pc.set_facecolor(color)
            axs[i].set_ylabel(f'Component {i+1}')
            axs[i].set_xticks(range(0,num_digits), range(num_digits))
            axs[i].set_xticklabels(range(num_digits))
            axs[i].set_xlabel('Digit')
            axs[i].set_yticks([min_val,max_val])
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.clf()

