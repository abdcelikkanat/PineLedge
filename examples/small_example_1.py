import torch
from src.base import BaseModel
from src.construction import ConstructionModel
from src.learning import LearningModel
from datasets.datasets import Dataset
from src.animation import Animation
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import sys
import utils

# # A model of two nodes
# x0 = [[-1, 0], [1, 0]]
# v = [[[1, 0], [-1, 0]], [[-1, 0], [1, 0]], [[1, 0], [-1, 0]]]
# beta = [4., 4.]
# bins_rwidth = len(v)
# last_time = 6

# A mode of four nodes
x0 = [[-1, 0], [1, 0], [0, 1], [0, -1]]
v = [[[1, 0], [-1, 0], [0, -1], [0, 1]],
     [[-1, 0], [1, 0], [0, 1], [0, -1]],
     [[1, 0], [-1, 0], [0, -1], [0, 1]]]
beta = np.ones((4, ) ) * 4  #[3., 3., 3., 3.]
bins_rwidth = len(v)
last_time = 6

# Set some paremeters
nodes_num = len(x0)
dim = len(x0[0])
batch_size = 64
learning_rate = 0.1
epochs_num = 200
seed = 123
verbose = True
time_normalization = True
shuffle = False
viz = 1

x0 = torch.as_tensor(x0, dtype=torch.float)
v = torch.as_tensor(v, dtype=torch.float)
beta = torch.as_tensor(beta, dtype=torch.float)

filename = "small_example_1"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', "datasets", "synthetic", f"{filename}.pkl")

# Construct the artificial network and save
cm = ConstructionModel(x0=x0, v=v, beta=beta, bins_rwidth=bins_rwidth, last_time=last_time, seed=seed)
cm.save(dataset_path)

# Load the dataset
dataset = Dataset(dataset_path, time_normalization=time_normalization)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=utils.collate_fn)
# Run the model
# print(dataset.get_last_time())
lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=len(v), dim=dim, last_time=1,
                   learning_rate=learning_rate, epochs_num=epochs_num, verbose=verbose, seed=seed)
lm.learn()
print(lm.get_model_params())
print(lm.get_bins_bounds()*6)

# lm.prediction(node_idx=torch.as_tensor([0, 1, 2, 3]),
#               event_times=torch.as_tensor([1.05, 1.1, 1.15, 1.2]),
#               test_middle_point=torch.as_tensor([1.1]), cholesky=True)

viz = 1
if viz == 1:
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
                     padding=0.1
                     )

    # Prediction animation
    embs_pred = lm.get_xt(times_list=times_list/float(last_time)).detach().numpy().reshape(-1, 2)
    anim = Animation(embs=embs_pred, time_list=node_times, group_labels=node_ids,
                     colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
                     title="Prediction model"+" ".join(filename.split('_')),
                     padding=0.1
                     )
