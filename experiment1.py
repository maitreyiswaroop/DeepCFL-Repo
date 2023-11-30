# One-One Experiment
# Imports
import os
import numpy as np
from sklearn import metrics as skmetrics
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy
from tqdm import tqdm
from matplotlib import pyplot as plt

from hyperparams import *
from data import get_MNIST_one_one_data
from models import Combined_VAE
from objectives import get_vae_loss, get_yh_pred_loss, get_auxiliary_classifier_loss, get_sparsity_loss
from data_visualisers import plot_images, violin_plts
from schedulers import SigmoidScheduler, CosineScheduler

import datetime

# getting the dataset
x_train, y_train, class_train, train_labels, x_test, y_test, class_test, test_labels, idx_sample, sample_digits=get_MNIST_one_one_data()

print("="*10)
print(f"Softmax={with_softmax} | scheduler={scheduler} | lambda_range={lambda_range}")
# Resetting the seed for each experiment
torch.manual_seed(torch_seed)
np.random.seed(np_seed)
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if fxy_type=='Identity':
    MLP_layers=None
else:
    MLP_layers=[]

model = Combined_VAE(x_l_shape=[1,28,28],y_l_shape=[1,28,28], dim_x_h=dim_x_h, dim_y_h=dim_y_h,
                x_encoder_specs=x_encoder_specs, x_decoder_specs=x_decoder_specs,
                y_encoder_specs=x_encoder_specs, y_decoder_specs=x_decoder_specs,
                MLP_layers=MLP_layers, softmax=with_softmax).to(device)

# We also train an auxiliary classifier to measure the quality of clustering for each class
auxiliary_classifier_x = nn.Linear(dim_x_h, dim_x_h).to(device)
auxiliary_classifier_y = nn.Linear(dim_y_h, dim_y_h).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer_aux = torch.optim.Adam(list(auxiliary_classifier_x.parameters()) + list(auxiliary_classifier_y.parameters()), lr=1e-3)

# lambda weight scheduler for prediction loss
if scheduler:
    sched_init = lambda_range[0]; sched_final = lambda_range[1]; sched_width = 10.0; sched_start = -0.2
    if scheduler=="Sigmoid":
        pred_loss_lambda_sched = SigmoidScheduler(init_val=sched_init, final_val=sched_final, width=sched_width, start=sched_start)
    elif scheduler=="Cosine":
        pred_loss_lambda_sched = CosineScheduler(init_val=sched_init,final_val=sched_final,period= sched_width)
    pred_loss_lambda = pred_loss_lambda_sched.init_val
else:
    pred_loss_lambda = lambda_range[0]

metrics = {
    "train_aux_accuracy": [],
    "test_aux_accuracy": [],
    "train_recon_loss": [],
    "test_recon_loss": [],
    "train_prior_loss": [],
    "test_prior_loss": [],
    "train_yh_pred_loss": [],
    "test_yh_pred_loss": [],
    "train_l1_loss": [],
    "test_l1_loss": [],
}

# checking the results folder for the most recent run number
if os.path.exists(f'./{results_dir}'):
    run_no = len(os.listdir(f'./{results_dir}'))+1
else:
    os.mkdir(f'./{results_dir}')
    run_no = 1

# creating the directory for the current run
os.mkdir(f'./{results_dir}/run_{run_no}')
cur_dir=f'./{results_dir}/run_{run_no}'
os.mkdir(f'{cur_dir}/nets')
os.mkdir(f'{cur_dir}/violins')
os.mkdir(f'{cur_dir}/tsne')
os.mkdir(f'{cur_dir}/reconstructions')

# Training
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_metrics = {k: [] for k in metrics.keys()}

    for i in tqdm(range(0, len(x_train), bs)):
        xl, yl, label = (
            x_train[i : i + bs].to(device),
            y_train[i : i + bs].to(device),
            train_labels[i : i + bs].to(device),
        )
        
        (x_recon, xh, mu_xh, logvar_xh), (y_recon, yh, mu_yh, logvar_yh), yh_prime = model(xl, yl)

        # Model training
        vae_loss_x, recon_loss_x, prior_loss_x = get_vae_loss(
            xl, x_recon, mu_xh, logvar_xh, beta=beta_x
        )
        vae_loss_y, recon_loss_y, prior_loss_y = get_vae_loss(
            yl, y_recon, mu_yh, logvar_yh, beta=beta_y
        )
        yh_pred_loss = get_yh_pred_loss(yh_prime, yh)
        # L1 loss for f_xy map
        l1_loss = get_sparsity_loss(model.f_xy, lambda2=1.0, d=l1_sum_dim)
        total_loss = vae_loss_x + vae_loss_y + pred_loss_lambda * yh_pred_loss + lambda_2 * l1_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Auxiliary classifier training
        # (.detach() is not actually necessary here)
        x_label_pred = auxiliary_classifier_x(xh.detach())
        y_label_pred = auxiliary_classifier_y(yh.detach())
        x_aux_loss = get_auxiliary_classifier_loss(x_label_pred, label)
        y_aux_loss = get_auxiliary_classifier_loss(y_label_pred, label)
        total_aux_loss = x_aux_loss + y_aux_loss
        optimizer_aux.zero_grad()
        total_aux_loss.backward()
        optimizer_aux.step()

        # Metrics
        x_label_pred = F.softmax(x_label_pred, dim=1)
        y_label_pred = F.softmax(y_label_pred, dim=1)
        label_accuracy = (
            accuracy(x_label_pred, label, task="multiclass", num_classes=10)
            + accuracy(y_label_pred, label, task="multiclass", num_classes=10)
        ) / 2
        epoch_metrics["train_aux_accuracy"].append(label_accuracy.item())
        epoch_metrics["train_recon_loss"].append(recon_loss_x.item())
        epoch_metrics["train_prior_loss"].append(prior_loss_x.item())
        epoch_metrics["train_yh_pred_loss"].append(yh_pred_loss.item())
        epoch_metrics["train_l1_loss"].append(l1_loss.item())

    for i in range(0, len(x_test), bs):
        xl, yl, label = (
            x_test[i : i + bs].to(device),
            y_test[i : i + bs].to(device),
            test_labels[i : i + bs].to(device),
        )

        model.eval()
        auxiliary_classifier_x.eval()
        auxiliary_classifier_y.eval()
        with torch.no_grad():
            (x_recon, xh, mu_xh, logvar_xh), (y_recon, yh, mu_yh, logvar_yh), yh_prime = model(xl, yl)
        model.train()
        auxiliary_classifier_x.train()
        auxiliary_classifier_y.train()

        # Metrics
        x_label_pred = auxiliary_classifier_x(xh)
        y_label_pred = auxiliary_classifier_y(yh)
        x_label_pred = F.softmax(x_label_pred, dim=1)
        y_label_pred = F.softmax(y_label_pred, dim=1)
        label_accuracy = (
            accuracy(x_label_pred, label, task="multiclass", num_classes=10)
            + accuracy(y_label_pred, label, task="multiclass", num_classes=10)
        ) / 2
        _, recon_loss_x, prior_loss_x = get_vae_loss(
            xl, x_recon, mu_xh, logvar_xh
        )
        _, recon_loss_y, prior_loss_y = get_vae_loss(
            yl, y_recon, mu_yh, logvar_yh
        )
        yh_pred_loss = get_yh_pred_loss(yh_prime, yh)
        l1_loss = get_sparsity_loss(model.f_xy, lambda2=1.0, d=l1_sum_dim)
        epoch_metrics["test_aux_accuracy"].append(label_accuracy.item())
        epoch_metrics["test_recon_loss"].append(recon_loss_x.item())
        epoch_metrics["test_prior_loss"].append(prior_loss_x.item())
        epoch_metrics["test_yh_pred_loss"].append(yh_pred_loss.item())
        epoch_metrics["test_l1_loss"].append(l1_loss.item())
    if scheduler:
        pred_loss_lambda = pred_loss_lambda_sched.step()    
    for k, v in epoch_metrics.items():
        metrics[k].append(sum(v) / len(v))
    # checking for convergence
    if epoch>15:
        # prediction loss convergence if the difference between the last two values is less than 0.1%
        if np.abs(metrics["train_yh_pred_loss"][-1]-metrics["train_yh_pred_loss"][-2])<1e-3:
            if np.abs(metrics["train_aux_accuracy"][-1]-metrics["train_aux_accuracy"][-2])<1e-3:
                if np.abs(metrics["test_aux_accuracy"][-1]-metrics["test_aux_accuracy"][-2])<1e-3:
                    print("Converged")
                    break

metric_names = ["aux_accuracy", "yh_pred_loss", "recon_loss", "prior_loss", "l1_loss"]
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for ax, metric_name in zip(axs.flatten(), metric_names):
    ax.plot(metrics[f"train_{metric_name}"], label=f"train")
    ax.plot(metrics[f"test_{metric_name}"], label=f"test")
    ax.set(xlabel="Epoch", ylabel=metric_name)
    if "accuracy" in metric_name:
        ax.set_ylim(0, 1)
    ax.legend()
fig.tight_layout()
fig.savefig(f"{cur_dir}/curves_lambda2={lambda_2}.png")
# plt.show()
plt.clf()

model.to("cpu")
model.eval()

(x_recon, xh, mu_xh, logvar_xh), (y_recon, yh, mu_yh, logvar_yh), yh_prime = model(x_test, y_test)
## Results
y_h_l_recon=model.AE_y.decoder(yh_prime[idx_sample]).detach()
plot_images(x_recon[idx_sample].detach(),test_labels[idx_sample], save=f"{cur_dir}/reconstructions/recon_x.png", fig_size=(10,1))
plot_images(y_recon[idx_sample].detach(),test_labels[idx_sample], save=f"{cur_dir}/reconstructions/recon_y.png", fig_size=(10,1))
plot_images(y_h_l_recon,test_labels[idx_sample], save=f"{cur_dir}/reconstructions/recon_yh_prime.png", fig_size=(10,1))
# for sample_dig in sample_digits:
y_h_l_recon=model.AE_y.decoder(yh_prime[sample_digits]).detach()
plot_images(torch.concat((y_test[sample_digits],y_h_l_recon),dim=0),torch.concat((test_labels[sample_digits],test_labels[sample_digits])),save=f"{cur_dir}/reconstructions/recon_yh_prime_all_digits.png",fig_size=(10,20))

y_h_recon=model.AE_y.decoder(yh[sample_digits]).detach()
plot_images(torch.concat((y_test[sample_digits],y_h_recon),dim=0),torch.concat((test_labels[sample_digits],test_labels[sample_digits])),save=f"{cur_dir}/reconstructions/recon_yh_all_digits.png",fig_size=(10,20))

x_h_recon=model.AE_x.decoder(xh[sample_digits]).detach()
plot_images(torch.concat((x_test[sample_digits],x_h_recon),dim=0),torch.concat((test_labels[sample_digits],test_labels[sample_digits])),save=f"{cur_dir}/reconstructions/recon_xh_all_digits.png",fig_size=(10,20))
# store silhouette scores in a file
with open(f"{cur_dir}/silhouette_scores.txt", "w") as f:
    sil_xh = 0.0; sil_yh=0.0; sil_yh_prime=0.0
    with torch.no_grad():
        print("Silhouette Scores")
        sil_xh += skmetrics.silhouette_score(X=xh.detach().numpy(),labels=test_labels)#, random_state=rand_state)
        sil_yh += skmetrics.silhouette_score(X=yh.detach().numpy(),labels=test_labels)#, random_state=rand_state)
        sil_yh_prime += skmetrics.silhouette_score(X=yh_prime.detach().numpy(),labels=test_labels)#, random_state=rand_state)
        print(f"xh: {sil_xh}")
        print(f"yh: {sil_yh}")
        print(f"yh_prime: {sil_yh_prime}")
        f.write(f"xh: {sil_xh}\n"); f.write(f"yh: {sil_yh}\n"); f.write(f"{cur_dir}/yh_prime: {sil_yh_prime}\n")

violin_plts(xh,test_labels.detach(), save_path=f"{cur_dir}/violins/violin_xh.png")
violin_plts(yh,test_labels.detach(), save_path=f"{cur_dir}/violins/violin_yh.png")
violin_plts(yh_prime,test_labels.detach(), save_path=f"{cur_dir}/violins/violin_yh_prime.png")
# saving the model
torch.save(model.state_dict(), f"{cur_dir}/nets/model.pt")
torch.save(auxiliary_classifier_x.state_dict(), f"{cur_dir}/nets/aux_classifier_x.pt")
torch.save(auxiliary_classifier_y.state_dict(), f"{cur_dir}/nets/aux_classifier_y.pt")

# TSNE plots
with torch.no_grad():
    tsne_xh = TSNE(n_components=2).fit_transform(xh.detach().numpy())
    tsne_yh = TSNE(n_components=2).fit_transform(yh.detach().numpy())
    tsne_yh_prime = TSNE(n_components=2).fit_transform(yh_prime.detach().numpy())
    # will need to reduce dimensions of x_test and y_test to 2
    tsne_x = TSNE(n_components=2).fit_transform(x_test.reshape(x_test.shape[0],-1).detach().numpy())
    tsne_y = TSNE(n_components=2).fit_transform(y_test.reshape(y_test.shape[0],-1).detach().numpy())

# x
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].set_title("x")
axes[0].set_xticks(np.arange(min(tsne_x[:,0])-1, max(tsne_x[:,0])+1, 5))
axes[0].set_yticks(np.arange(min(tsne_x[:,1])-1, max(tsne_x[:,1])+1, 5))
for i in range(10):
    axes[0].scatter(tsne_x[:,0][test_labels==i],tsne_x[:,1][test_labels==i],c=colors[i])
# axes[0].legend([f"digit {i}" for i in range(10)])

axes[1].set_title("xh")
axes[1].set_xticks(np.arange(min(tsne_xh[:,0]), max(tsne_xh[:,0])+1, 5))
axes[1].set_yticks(np.arange(min(tsne_xh[:,1]), max(tsne_xh[:,1])+1, 5))
for i in range(10):
    axes[1].scatter(tsne_xh[:,0][test_labels==i],tsne_xh[:,1][test_labels==i],c=colors[i])
# axes[1].legend([f"digit {i}" for i in range(10)])
plt.legend([f"digit {i}" for i in range(10)])
plt.savefig(f"{cur_dir}/tsne/tsne_xh.png")
# Clear the plots
for ax in axes:
    ax.clear()
plt.close()

# y
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].set_title("y")
axes[0].set_xticks(np.arange(min(tsne_y[:,0])-1, max(tsne_y[:,0])+1, 5))
axes[0].set_yticks(np.arange(min(tsne_y[:,1])-1, max(tsne_y[:,1])+1, 5))
for i in range(10):
    axes[0].scatter(tsne_y[:,0][test_labels==i],tsne_y[:,1][test_labels==i],c=colors[i])
# axes[0].legend([f"digit {i}" for i in range(10)])

axes[1].set_title("yh")
axes[1].set_xticks(np.arange(min(tsne_yh[:,0])-1, max(tsne_yh[:,0])+1, 5))
axes[1].set_yticks(np.arange(min(tsne_yh[:,1])-1, max(tsne_yh[:,1])+1, 5))
for i in range(dim_y_h):
    # axes[1].scatter(tsne_yh[:,0][test_labels==i],tsne_yh[:,1][test_labels==i],c=colors[i])
    axes[1].scatter(tsne_yh[:,0][test_labels==i],tsne_yh[:,1][test_labels==i],c=colors[i])
# axes[1].legend([f"digit {i}" for i in range(10)])
# single legend for both plots
plt.legend([f"digit {i}" for i in range(10)])
plt.savefig(f"{cur_dir}/tsne/tsne_yh.png")
# Clear the plots
for ax in axes:
    ax.clear()
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(10, 10))
axes.set_title("yh_prime")
axes.set_xticks(np.arange(min(tsne_yh_prime[:,0])-1, max(tsne_yh_prime[:,0])+1, 5))
axes.set_yticks(np.arange(min(tsne_yh_prime[:,1])-1, max(tsne_yh_prime[:,1])+1, 5))
for i in range(dim_y_h):
    # axes.scatter(tsne_yh_prime[:,0][test_labels==i],tsne_yh_prime[:,1][test_labels==i],c=colors[i])
    axes.scatter(tsne_yh_prime[:,0][test_labels==i],tsne_yh_prime[:,1][test_labels==i],c=colors[i]) 
# legend to the right of the plot
plt.legend([f"digit {i}" for i in range(10)],bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f"{cur_dir}/tsne/tsne_yh_prime.png")
# Clear the plots
axes.clear()
plt.close()

# f_xy
with torch.no_grad():
    with open(f"{cur_dir}/f_xy.txt", "w") as f:
        f.write(f"f_xy\n")
        for param in model.f_xy.parameters():
            f.write(f"{param}\n")
        # write model
        f.write(f"model\n")
        f.write(f"{model}\n")

# writing all the hyperparameters to a file
with open(f"{cur_dir}/hyperparameters.txt", "w") as f:
    f.write(f"Run on {current_date}\n")
    f.write(f"torch_seed: {torch_seed} | numpy: {np_seed} \n")
    f.write(f"fxy_type: {fxy_type}\n")
    f.write(f"HYPERPARAMETERS:\n\tpred_loss_lambda: {pred_loss_lambda}\n")
    f.write(f"\tbeta_x: {beta_x}\n")
    f.write(f"\tbeta_y: {beta_y}\n")
    f.write(f"\tlambda_2: {lambda_2}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"batch_size: {bs}\n")
    f.write(f"optimizer: Adam\n")
    f.write(f"learning_rate: {lr}\n")
    if separate_optimizers:
        print("Separate optimizers")
        f.write(f"learning_rate_fxy:{lr_fxy}\n")
        if alt_opt_update:
            print("Alternate optimizer update")
            f.write(f"alt_opt_update: True, freq: {alt_opt_update_freq}\n")
    if scheduler:
        f.write(f"Scheduler:{scheduler}\n")
        f.write(f"\t sched_init:{sched_init} | sched_final:{sched_final}\n\tsched_width:{sched_width} | sched_start:{sched_start}")
        if scheduler=="Sigmoid":
                pred_loss_lambda_sched = SigmoidScheduler(init_val=sched_init, final_val=sched_final, width=sched_width, start=sched_start)
        elif scheduler=="Cosine":
            pred_loss_lambda_sched = CosineScheduler(init_val=sched_init,final_val=sched_final,period= sched_width)
        pred_loss_lambda_sched.visualize(num_steps=epochs, save=f'{cur_dir}/scheduler.png')
    f.write(f"lambda_2: {lambda_2}\n")
