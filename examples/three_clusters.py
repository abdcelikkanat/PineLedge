import torch
from src.base import BaseModel
from src.construction import ConstructionModel
from src.learning import LearningModel
from datasets.sil_datasets import Dataset
from torch.utils.data import DataLoader
from src.experiments import Experiments
from src.events import Events
import numpy as np
from src.estimation import Estimation
import pandas as pd
import os
import sys
import utils
import pickle
from sklearn.metrics import roc_auc_score

# A mode of four nodes
# x0 = [[-5, 0], [5, 0], [0, 5], [0, -5]]
# v = [[[1, 0], [-1, 0], [0, -1], [0, 1]],
#      [[-1, 0], [1, 0], [0, 1], [0, -1]],
#      [[1, 0], [-1, 0], [0, -1], [0, 1]]]

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
beta = np.ones((1, ) ) * 1.05  #[3., 3., 3., 3.]
bins_rwidth = v.shape[0]
last_time = sum(time_interval_lengths)

# Set some paremeters
nodes_num = x0.shape[0]
dim = x0.shape[1]
batch_size = 45 #1
learning_rate = 0.1
epochs_num = 400  # 500
steps_per_epoch = 5
seed = 34440
verbose = True
time_normalization = True
shuffle = True
learn = False
visualization = True
experiment = False

x0 = torch.as_tensor(x0, dtype=torch.float)
v = torch.as_tensor(v, dtype=torch.float)
beta = torch.as_tensor(beta, dtype=torch.float)

dataset = "three_clusters"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', "datasets", "synthetic", f"{dataset}_events.pkl")

# Construct the artificial network and save
if not os.path.exists(dataset_path):
    cm = ConstructionModel(x0=x0, v=v, beta=beta, bins_rwidth=bins_rwidth, last_time=last_time, seed=seed)
    cm.save(dataset_path)

# Load the dataset
events = Events(seed=seed, batch_size=batch_size)
events.read(dataset_path)
events.normalize(init_time=0, last_time=1.0)
data_loader = DataLoader(events, batch_size=1, shuffle=shuffle, collate_fn=utils.collate_fn)

# Run the model
lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=len(v), dim=dim, last_time=1.,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch, verbose=verbose,
                   seed=seed)

if learn:
    lm.learn()
    # Save the model
    torch.save(lm.state_dict(), "./three_clusters.model")
else:
    lm.load_state_dict(torch.load("./three_clusters.model"))

print("Bins", lm.get_bins_bounds()*last_time)

if visualization:
    # Visualization
    from visualization.animation import Animation
    times_list = torch.linspace(0, 1.0, 100)
    # Ground truth animation
    bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_rwidth=bins_rwidth)
    embs_gt = bm.get_xt(times_list=times_list*last_time).detach().numpy()
    node2color = [0] * cluster_sizes[0] + [1] * cluster_sizes[1] + [2] * cluster_sizes[2]
    anim = Animation(embs_gt, fps=12, node2color=node2color)
    anim.save("./three_clusters_gt.mp4")

    embs_pred = lm.get_xt(times_list=times_list).detach().numpy()
    node2color = [0] * cluster_sizes[0] + [1] * cluster_sizes[1] + [2] * cluster_sizes[2]
    anim = Animation(embs_pred, fps=12, node2color=node2color)
    anim.save("./three_clusters_pm.mp4")

if experiment:

    est = Estimation(lm=lm, init_time=0.0, split_time=1.0, last_time=1.0)

    sample_dict_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         f"../experiments/samples/{dataset}_tInit=0.0_tLast=1.0_sTime=1.0_binNum=20_subsampling=0_dict.pkl")
    with open(sample_dict_file_path, 'rb') as f:
        sample_file = pickle.load(f)
    train_samples, train_labels = sample_file["train"]["samples"], sample_file["train"]["labels"]

    labels = train_labels
    samples = train_samples
    y_pred = []
    y_true = []
    for i, j in zip(*np.triu_indices(nodes_num, k=1)):

        if len(labels[i][j]):

            # print(samples[i][j])
            y_pred_ij = est.get_log_intensity(time_list=samples[i][j], node_pairs=[[i], [j]]).tolist()
            y_pred.extend(y_pred_ij)

            y_true.extend(labels[i][j])
            # assert len(y_true) == len(y_pred), f"OLMAZ {len(y_true)} == {len(y_pred)}. {y_pred_ij}"

        else:
            pass  # print(i,j)

    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    print(f"Train AUC: {auc}")

    F = events.get_freq()
    F_sum = float(F.sum())
    events_data = events.get_data()
    baseline_nll = 0
    for i, j in zip(*np.triu_indices(nodes_num, k=1)):
        if len(events_data[i][j]):
            baseline_nll += -np.log( F[i][j] / F_sum ) * len(events_data[i][j])

    print(f"Baseline neg. log-likelihood: {baseline_nll}")
