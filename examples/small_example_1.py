import torch
from src.base import BaseModel
from src.construction import ConstructionModel
from src.learning import LearningModel
from datasets.datasets import Dataset
from src.animation import Animation
from torch.utils.data import DataLoader
from src.experiments import Experiments
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
x0 = [[-5, 0], [5, 0], [0, 5], [0, -5]]
v = [[[1, 0], [-1, 0], [0, -1], [0, 1]],
     [[-1, 0], [1, 0], [0, 1], [0, -1]],
     [[1, 0], [-1, 0], [0, -1], [0, 1]]]
beta = np.ones((len(x0), ) ) * 1.05  #[3., 3., 3., 3.]
bins_rwidth = len(v)
last_time = 30

# Set some paremeters
nodes_num = len(x0)
dim = len(x0[0])
batch_size = 4
learning_rate = 0.01
epochs_num = 300
seed = 123
verbose = True
time_normalization = True
shuffle = True
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
dataset = Dataset(dataset_path, init_time=0, last_time=last_time, shuffle=shuffle, time_normalization=time_normalization)
dataset.plot_events(node_pairs=[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=utils.collate_fn)
# Run the model
lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=len(v), dim=dim, last_time=1.,
                   learning_rate=learning_rate, epochs_num=epochs_num, verbose=verbose, seed=seed)
lm.learn()
print(lm.get_model_params())
print(lm.get_bins_bounds()*last_time)


if viz == 1:
    # Visualization
    times_list = torch.linspace(0, last_time, 100)
    node_ids = np.tile(np.arange(nodes_num).reshape(1, -1), (len(times_list), 1)).flatten()
    node_times = np.tile(np.asarray(times_list).reshape(-1, 1), (1, nodes_num)).flatten()
    colors = ['r', 'b', 'g', 'm']
    # # Ground truth animation
    # bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_rwidth=bins_rwidth)
    # embs_gt = bm.get_xt(times_list=times_list).detach().numpy().reshape(-1, 2)
    # anim = Animation(embs=embs_gt, time_list=node_times, group_labels=node_ids,
    #                  colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
    #                  title="Ground-truth model"+" ".join(filename.split('_')),
    #                  padding=0.1
    #                  )

    # Prediction animation
    # print(times_list)
    embs_pred = lm.get_xt(times_list=times_list/last_time).detach().numpy()
    anim = Animation(embs=embs_pred, time_list=node_times, group_labels=node_ids,
                     colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
                     title="Prediction model"+" ".join(filename.split('_')),
                     padding=0.1,
                     dataset=None,
                     figure_path=f"../outputs/4nodes_rbf+periodic.html"
                     )


# result = lm.prediction(event_times=torch.as_tensor([1.05, 1.1, 1.15, 1.2, 1.3, 1.1]),
#                        event_node_pairs=torch.as_tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]).transpose(0, 1),
#                        test_middle_point=torch.as_tensor([1.5]), cholesky=True)
# print(result)

# Experiments
from sklearn.metrics import roc_auc_score

exp = Experiments(dataset=dataset, set_type="train")
exp.plot()
samples, labels = exp.get_samples(), exp.get_labels()
F = exp.get_freq_map()
simon_auc = roc_auc_score(y_true=labels, y_score=[F[sample[0], sample[1]] for sample in samples])
print(f"Simon auc score: {simon_auc}")

model_params = lm.get_model_params()
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