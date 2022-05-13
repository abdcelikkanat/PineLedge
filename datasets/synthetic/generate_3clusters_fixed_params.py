import os
import sys
import torch
import numpy as np
import pickle as pkl
import utils
from src.construction import ConstructionModel


time_interval_lengths = [10., 10., 10.]
cluster_sizes = [15, 20, 10]
x0_c1 = np.random.multivariate_normal(mean=[-5, 5], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
x0_c2 = np.random.multivariate_normal(mean=[5, 5], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
x0_c3 = np.random.multivariate_normal(mean=[0, -5], cov=np.eye(2, 2), size=(cluster_sizes[2], ))
x0 = np.vstack((np.vstack((x0_c1, x0_c2)), x0_c3))

x1_c1 = np.random.multivariate_normal(mean=[5, 5], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
x1_c2 = np.random.multivariate_normal(mean=[-5, 5], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
x1_c3 = np.random.multivariate_normal(mean=[5, 5], cov=np.eye(2, 2), size=(cluster_sizes[2], ))
x1 = np.vstack((np.vstack((x1_c1, x1_c2)), x1_c3))
v1 = (x1 - x0) / time_interval_lengths[0]

x2_c1 = np.random.multivariate_normal(mean=[0, -5], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
x2_c2 = np.random.multivariate_normal(mean=[5, 5], cov=np.eye(2, 2), size=(cluster_sizes[1], ))
x2_c3 = np.random.multivariate_normal(mean=[-5, 5], cov=np.eye(2, 2), size=(cluster_sizes[2], ))
x2 = np.vstack((np.vstack((x2_c1, x2_c2)), x2_c3))
v2 = (x2 - x1) / time_interval_lengths[1]

x3_c1 = np.random.multivariate_normal(mean=[-5, 5], cov=np.eye(2, 2), size=(cluster_sizes[0], ))
x3_c2 = np.random.multivariate_normal(mean=[5, 5], cov=2*np.eye(2, 2), size=(cluster_sizes[1], ))
x3_c3 = np.random.multivariate_normal(mean=[0, -5], cov=np.eye(2, 2), size=(cluster_sizes[2], ))
x3 = np.vstack((np.vstack((x3_c1, x3_c2)), x3_c3))
v3 = (x3 - x2) / time_interval_lengths[1]

v = np.vstack((np.vstack(([v1], [v2])), [v3]))
beta_coeff = 1
beta = np.ones((sum(cluster_sizes), ) ) * beta_coeff
bins_num = v.shape[0]
last_time = sum(time_interval_lengths)


x0 = torch.as_tensor(x0, dtype=torch.float)
v = torch.as_tensor(v, dtype=torch.float)
beta = torch.as_tensor(beta, dtype=torch.float)

# Set the parameters
verbose = True
suffix = f"_beta={beta_coeff}"
seed = 0

# Define the folder and dataset paths
dataset_name = f"three_clusters_fp_sizes="+"_".join(map(str, cluster_sizes))+suffix
dataset_folder = os.path.join(
    utils.BASE_FOLDER,
    "datasets", "synthetic", dataset_name
)
node2group_path = os.path.join(
    dataset_folder, f"{dataset_name}_node2group.pkl"
)

# Check if the file exists
assert not os.path.exists(dataset_folder), "This folder path exists!"

# Create the folder of the dataset
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Construct the artificial network and save
cm = ConstructionModel(x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=last_time, verbose=verbose, seed=seed)
cm.save(dataset_folder)

# Group labels
node2group_data = {
    "node2group": {sum(cluster_sizes[:c_idx])+idx: c_idx for c_idx, c_size in enumerate(cluster_sizes) for idx in range(c_size)},
    "group2node": {c_idx: [sum(cluster_sizes[:c_idx])+idx for idx in range(c_size)] for c_idx, c_size in enumerate(cluster_sizes)}
}
with open(node2group_path, "wb") as f:
    pkl.dump(node2group_data, f)

