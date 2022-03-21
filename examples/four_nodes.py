import torch
from src.base import BaseModel
from src.construction import ConstructionModel
from src.learning import LearningModel
from datasets.sil_datasets import Dataset
from src.animation import Animation
from torch.utils.data import DataLoader
from src.experiments import Experiments
from src.events import Events
import numpy as np
from src.estimation import Estimation
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
x0 = [[-5, 0], [5, 0], [0, 5], [0, -5]]
v = [[[1, 0], [-1, 0], [0, -1], [0, 1]],
     [[-1, 0], [1, 0], [0, 1], [0, -1]],
     [[1, 0], [-1, 0], [0, -1], [0, 1]]]
beta = np.ones(( len(x0), ) ) * 1.05  #[3., 3., 3., 3.]
bins_rwidth = len(v)
last_time = 30

# Set some paremeters
nodes_num = len(x0)
dim = len(x0[0])
batch_size = 4 #1
learning_rate = 0.1
epochs_num = 100  # 500
steps_per_epoch = 100
seed = 123
verbose = True
time_normalization = True
shuffle = True
visualization = True

x0 = torch.as_tensor(x0, dtype=torch.float)
v = torch.as_tensor(v, dtype=torch.float)
beta = torch.as_tensor(beta, dtype=torch.float)

filename = "4nodedense_events"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', "datasets", "synthetic", f"{filename}.pkl")

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
lm.learn()

print("Bins", lm.get_bins_bounds()*last_time)

est = Estimation(lm=lm, init_time=0., split_time=1.0, last_time=1.)


if visualization:
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
    # print(times_list)

    # embs_pred = lm.get_xt(times_list=times_list/last_time).detach().numpy()
    # anim = Animation(embs=embs_pred, time_list=node_times, group_labels=node_ids,
    #                  colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
    #                  title="Predicted model", #title="Predicted model"+" ".join(filename.split('_')),
    #                  padding=0.1,
    #                  dataset=None,
    #                  # figure_path=f"../experiments/outputs/4nodes.html"
    #                  )

# Experiments
from sklearn.metrics import roc_auc_score

exp = Experiments(seed=seed)
exp.set_events(events=events.get_train_events(last_time=1.0))
labels, samples = exp.construct_samples(bins_num=5, subsampling=0, with_time=True, init_time=0.0, last_time=1.0)

exp.plot_samples(labels=labels, samples=samples)

F = exp.get_freq_map()
simon_auc = roc_auc_score(y_true=labels, y_score=[F[sample[0], sample[1]] for sample in samples])
print(f"Simon auc score: {simon_auc}")

# model_params = lm.get_model_params()
distances = []
for sample in samples:
    # print(sample[0:2])
    d = lm.get_pairwise_distances(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
    # distances.append(torch.exp(model_params['beta'][sample[0]] + model_params['beta'][sample[1]] - d))
    # distances.append(torch.exp( - d))
    # distances.append(d)
    # distances.append(
    #     +lm.get_log_intensity(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
    #     +lm.get_intensity_integral(node_pairs=torch.as_tensor([sample[0:2]]).t())
    # )
    # distances.append(
    #     +lm.get_log_intensity(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
    #     -lm.get_intensity_integral(node_pairs=torch.as_tensor([sample[0:2]]).t())
    # )
    distances.append(
        lm.get_log_intensity(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
    )

pred_auc = roc_auc_score(y_true=labels, y_score=distances)
print(f"pred auc score: {pred_auc}")