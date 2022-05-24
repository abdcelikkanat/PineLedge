import os
import sys
import torch
import numpy as np
import pickle as pkl
import utils
from src.base import BaseModel
from src.construction import ConstructionModel
from visualization.animation import Animation


def _get_B_factor(prior_B_sigma, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor, only_kernel=False):

    time_mat = bin_centers1 - bin_centers2.transpose(1, 2)
    time_mat = time_mat.squeeze(0)

    B_sigma = utils.softplus(prior_B_sigma)
    kernel = torch.exp(-0.5 * torch.div(time_mat ** 2, B_sigma))

    # Add a constant term to get rid of computational problems
    # kernel = kernel + 10 * utils.EPS * torch.eye(n=kernel.shape[0], m=kernel.shape[1])

    if only_kernel:
        return kernel

    # B x B lower triangular matrix
    L = torch.linalg.cholesky(kernel)

    return L

def _get_C_factor(prior_C_Q):

    # K x N matrix
    return prior_C_Q #torch.softmax(prior_C_Q, dim=0)

def _get_D_factor(dim):
    # D x D matrix
    return torch.eye(dim)



time_interval_lengths = [10.] * 20
cluster_sizes = [4]*25 #[15, 20, 10]

dim = 2
nodes_num = sum(cluster_sizes)
bins_num = len(time_interval_lengths)

prior_lambda = 1e-1
prior_sigma = 0.2
prior_B_sigma = 1e+4 #1e-4

beta_coeff = 2.5

# Set the parameters
verbose = True
seed = 0

# Define the folder and dataset paths
dataset_name = f"{len(cluster_sizes)}_clusters_mg_B={bins_num}_noise_s={prior_sigma}_rbf-s={prior_B_sigma}_lambda={prior_lambda}_sizes="+"_".join(map(str, cluster_sizes))+f"_beta={beta_coeff}"
# dataset_folder = os.path.join(
#     utils.BASE_FOLDER,
#     "datasets", "synthetic", dataset_name
# )
dataset_folder = os.path.join("/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/", dataset_name)
node2group_path = os.path.join(
    dataset_folder, f"{dataset_name}_node2group.pkl"
)


# x0_c1 = np.random.multivariate_normal(mean=[-5, 5], cov=np.eye(dim, dim), size=(cluster_sizes[0], ))
# x0_c2 = np.random.multivariate_normal(mean=[5, 5], cov=np.eye(dim, dim), size=(cluster_sizes[1], ))
# x0_c3 = np.random.multivariate_normal(mean=[0, -5], cov=np.eye(dim, dim), size=(cluster_sizes[2], ))
# x0 = np.vstack((np.vstack((x0_c1, x0_c2)), x0_c3))
x0 = np.empty(shape=(0, dim))
for c in range(len(cluster_sizes)):
    cluster_mean = np.random.multivariate_normal(mean=np.zeros(shape=(dim, )), cov=36*np.eye(dim, dim), size=(1, ))[0]
    x0 = np.vstack(
        (x0, np.random.multivariate_normal(mean=cluster_mean, cov=0.1*np.eye(dim, dim), size=(cluster_sizes[c],)))
    )


K = len(cluster_sizes)
final_dim = nodes_num * bins_num * dim

bin_centers = 0.5 * torch.as_tensor(time_interval_lengths)
bin_centers[1:] += torch.cumsum(torch.as_tensor(time_interval_lengths[1:]), dim=0)
bin_centers = bin_centers.view(1, 1, len(time_interval_lengths))
B_factor = _get_B_factor(
    prior_B_sigma=torch.as_tensor(prior_B_sigma), bin_centers1=bin_centers, bin_centers2=bin_centers, only_kernel=True
)

prior_C_Q = torch.zeros(size=(K, sum(cluster_sizes)), dtype=torch.float)
for k in range(K):
    prior_C_Q[k, range(sum(cluster_sizes[:k]), sum(cluster_sizes[:k+1]))] = 1
'''
prior_C_Q[0, range(sum(cluster_sizes[:1]))] = 1
prior_C_Q[1, range(sum(cluster_sizes[:1]), sum(cluster_sizes[:2]))] = 1
prior_C_Q[2, range(sum(cluster_sizes[:2]), sum(cluster_sizes[:3]))] = 1
'''
C_factor = _get_C_factor(prior_C_Q).T

#C_factor = torch.ones_like(C_factor)
D_factor = torch.eye(n=dim)
cov_factor = (prior_lambda) * torch.kron(torch.kron(B_factor, C_factor.contiguous()), D_factor.contiguous())
cov_diag = (prior_lambda**2) * (prior_sigma**2) * torch.ones(final_dim)

lmn = torch.distributions.LowRankMultivariateNormal(
    loc=torch.zeros(size=(final_dim,)),
    cov_factor=cov_factor,
    cov_diag=cov_diag
)

sample = lmn.sample()
v = sample.reshape(shape=(len(time_interval_lengths), nodes_num, dim))
# v = utils.unvectorize(sample, size=(len(time_interval_lengths), nodes_num, dim))

beta = np.ones((sum(cluster_sizes), ) ) * beta_coeff
last_time = sum(time_interval_lengths)


x0 = torch.as_tensor(x0, dtype=torch.float)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(v[0, :, 0], v[0, :, 1], '.')
# plt.show()
v = torch.as_tensor(v, dtype=torch.float)
beta = torch.as_tensor(beta, dtype=torch.float)

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


# Ground truth animation
frame_times = torch.linspace(0, sum(time_interval_lengths), 100)
bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_num=bins_num)
embs_pred = bm.get_xt(
    events_times_list=torch.cat([frame_times]*bm.get_number_of_nodes()),
    x0=torch.repeat_interleave(bm.get_x0(), repeats=len(frame_times), dim=0),
    v=torch.repeat_interleave(bm.get_v(), repeats=len(frame_times), dim=1)
).reshape((bm.get_number_of_nodes(), len(frame_times),  bm.get_dim())).transpose(0, 1).detach().numpy()

# Construct the data
events_list, events_pairs = [], []
events_dict = cm.get_events()
for i, j in utils.pair_iter(n=nodes_num):
    events_pair = events_dict[i][j]
    if len(events_pair):
        events_list.append(events_pair)
        events_pairs.append([i, j])
data = events_pairs, events_list

# Read the group information
node2group_filepath = os.path.join(dataset_folder, f"{dataset_name}_node2group.pkl")
with open(node2group_filepath, "rb") as f:
    node2group_data = pkl.load(f)
node2group, group2node = node2group_data["node2group"], node2group_data["group2node"]
# Animate
node2color = [node2group[idx] for idx in range(nodes_num)]
anim = Animation(embs_pred, data=data, fps=12, node2color=node2color, frame_times=frame_times.numpy())
anim.save(os.path.join(dataset_folder, "gt_animation.mp4"))
# anim.save(os.path.join("/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/", "gt_animation.mp4"))
