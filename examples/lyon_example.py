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


# A mode of four nodes
# x0 = [[-1, 0], [1, 0], [0, 1], [0, -1]]
# v = [[[1, 0], [-1, 0], [0, 1], [0, -1]], [[-1, 0], [1, 0], [0, -3], [0, 3]], [[1, 0], [-1, 0], [0, 1], [0, -1]]]
# beta = [3., 3., 3., 3.]
# bins_width = len(v)
# last_time = 6

bins_width = 10
dim = 2
# nodes_num = len(x0)
# dim = len(x0[0])
batch_size = 64
learning_rate = 0.001
epochs_num = 5
seed = 123
verbose = True

#
# x0 = torch.as_tensor(x0, dtype=torch.float)
# v = torch.as_tensor(v, dtype=torch.float)
# beta = torch.as_tensor(beta, dtype=torch.float)


filename = "LyonSchool_events"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "datasets", "real", f"{filename}.pkl")
figure_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "outputs", f"{filename}.html")

# Load the dataset
dataset = Dataset(dataset_path, time_normalization=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_fn)

nodes_num = dataset.get_num_of_nodes()
last_time = dataset.get_last_time()

# Run the model
lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=bins_width, dim=dim, last_time=last_time,
                   learning_rate=learning_rate, epochs_num=epochs_num, verbose=verbose, seed=seed)
lm.learn()


# Visualization
times_list = torch.linspace(0, last_time, 100)
node_groups = np.asarray(dataset.get_groups()) if dataset.get_groups() is not None else np.arange(nodes_num)
node_groups = np.tile(node_groups.reshape(1, -1), (len(times_list), 1)).flatten()
node_times = np.tile(np.asarray(times_list).reshape(-1, 1), (1, nodes_num)).flatten()
# colors = ['r', 'b', 'g', 'm']

# Prediction animation
embs_pred = lm.get_xt(times_list=times_list).detach().numpy().reshape(-1, 2)
anim = Animation(embs=embs_pred, time_list=node_times, group_labels=node_groups,
                 colors=list(map(str, node_groups)), color_name="Nodes",
                 title="Prediction model"+" ".join(filename.split('_')),
                 padding=0.1,
                 figure_path=figure_path
                 )