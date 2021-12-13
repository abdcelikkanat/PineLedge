import torch
from src.base import BaseModel
from src.construction import ConstructionModel
from src.learning import LearningModel
from datasets.datasets import Dataset
from src.animation import Animation
from torch.utils.data import DataLoader
from datasets.synthetic.generate import ClusterGraph
import numpy as np
import pandas as pd
import os
import sys
import utils

bins_rwidth = [3, 5]
cluster_sizes = [4]
cg = ClusterGraph(cluster_sizes=cluster_sizes, bins_rwidth=bins_rwidth)
x0, v = cg.get_params()


beta = np.ones(shape=(x0.shape[0], ), dtype=np.float) * 4
last_time = sum(bins_rwidth)

# Set some paremeters
nodes_num = len(x0)
dim = len(x0[0])
batch_size = 64
learning_rate = 0.1
epochs_num = 300
seed = 123
verbose = True
time_normalization = False
shuffle = False
viz = 1

x0 = torch.as_tensor(x0, dtype=torch.float)
v = torch.as_tensor(v, dtype=torch.float)
beta = torch.as_tensor(beta, dtype=torch.float)

filename = f"cluster_example_n={nodes_num}"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', "datasets", "synthetic", f"{filename}.pkl")

# Construct the artificial network and save
cm = ConstructionModel(x0=x0, v=v, beta=beta, bins_rwidth=bins_rwidth, last_time=last_time, seed=seed)
cm.save(dataset_path)

# Load the dataset
dataset = Dataset(dataset_path, time_normalization=time_normalization)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=utils.collate_fn)

lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=3, dim=dim, last_time=last_time,
                   learning_rate=learning_rate, epochs_num=epochs_num, verbose=verbose, seed=seed)
lm.learn()

# Visualization
times_list = torch.linspace(0, last_time, 100)
node_ids = np.tile(np.arange(nodes_num).reshape(1, -1), (len(times_list), 1)).flatten()
node_times = np.tile(np.asarray(times_list).reshape(-1, 1), (1, nodes_num)).flatten()
colors = ['r', 'b', 'g', 'm']
# Ground truth animation
bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_rwidth=bins_rwidth)
embs_gt = bm.get_xt(times_list=times_list).detach().numpy().reshape(-1, 2)
anim = Animation(embs=embs_gt, time_list=node_times, group_labels=node_ids,
                 colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
                 title="Ground-truth model"+" ".join(filename.split('_')),
                 padding=0.1, figure_path=f"../outputs/{filename}_groundtruth.html"
                 )

# Prediction animation
embs_pred = lm.get_xt(times_list=times_list).detach().numpy().reshape(-1, 2)
anim = Animation(embs=embs_pred, time_list=node_times, group_labels=node_ids,
                 colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
                 title="Prediction model"+" ".join(filename.split('_')),
                 padding=0.1, figure_path=f"../outputs/{filename}_pred.html"
                 )