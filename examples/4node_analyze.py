import torch
from src.base import BaseModel
from src.construction import ConstructionModel
from src.learning import LearningModel
from datasets.datasets import Dataset
from src.animation import Animation
from torch.utils.data import DataLoader
from src.experiments import Experiments
from src.prediction import PredictionModel
import numpy as np
import pandas as pd
import os
import sys
import utils.utils
import torch
from sklearn.metrics import average_precision_score
torch.set_num_threads(16)

# # Dataset name
# dataset_name = "ia_contacts" #"LyonSchool" #"resistance_game=4" ia_enron ia_contacts
# # Define the dataset path
# dataset_path = os.path.join(
#     os.path.dirname(os.path.realpath(__file__)), "..", "datasets", "real", dataset_name, f"{dataset_name}_events.pkl"
# )

dataset_name = "4nodedense"
# Define the dataset path
dataset_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "datasets", "synthetic", f"{dataset_name}_events.pkl"
)


# Set some parameters
dim = 2
bins_num = 10 # 10
batch_size = 2  # 8
learning_rate = 0.01 # 0.01
epochs_num = 400  # 400
seed = 123
verbose = True
time_normalization = True
shuffle = True
actions = ["learn", "expereval"]  # "learn",

# Define the model path
model_file_path = os.path.join(
    "../outputs/", f"{dataset_name}_lr={learning_rate}_batch={batch_size}_B={bins_num}_D={dim}_scale={time_normalization}_seed={seed}.model"
)

# Load the dataset
dataset = Dataset(dataset_path, init_time=0, time_normalization=time_normalization, train_ratio=1.0, )
print(f"Number of nodes: {dataset.get_num_of_nodes()}")

# Run the model
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=utils.collate_fn, drop_last=True)
lm = LearningModel(data_loader=data_loader, nodes_num=dataset.get_num_of_nodes(), bins_num=bins_num, dim=dim,
                   last_time=1.0, learning_rate=learning_rate, epochs_num=epochs_num,
                   verbose=verbose, seed=seed)


if "learn" in actions:

    lm.learn()

    # # Save the model
    torch.save(lm.state_dict(), model_file_path)

if "animate" in actions:

    # Load the model
    lm.load_state_dict(torch.load(model_file_path))

    # Define the figure path
    figure_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "outputs", f"{dataset_name}.html"
    )

    # Visualization
    times_list = torch.linspace(0, dataset.get_last_time(), 100)
    node_ids = np.tile(np.arange(dataset.get_num_of_nodes()).reshape(1, -1), (len(times_list), 1)).flatten()
    node_times = np.tile(np.asarray(times_list).reshape(-1, 1), (1, dataset.get_num_of_nodes())).flatten()
    colors = ['r', 'b', 'g', 'm']

    # Prediction animation
    embs_pred = lm.get_xt(times_list=times_list).detach().numpy()
    print(embs_pred.shape)
    anim = Animation(embs=embs_pred, time_list=node_times, group_labels=node_ids, dataset=None,
                     colors=[colors[id % len(colors)] for id in node_ids], color_name="Nodes",
                     title="Prediction model"+" ".join(dataset_name.split('_')),
                     padding=0.1,
                     # figure_path=f"../outputs/{dataset_name}_gauss+periodic.html"
                     )

# Experiments
if "expereval" in actions:
    from sklearn.metrics import roc_auc_score

    set_type = "train"
    # Load the model
    lm.load_state_dict(torch.load(model_file_path))

    exp = Experiments(dataset=dataset, set_type=set_type, num_of_bounds=5)
    samples, labels = exp.get_samples(), exp.get_labels()
    # utils.plot_events(num_of_nodes=dataset.get_num_of_nodes(), samples=samples, labels=labels, )

    t = torch.as_tensor([0.0])
    pm = PredictionModel(
        lm=lm, test_init_time=dataset.get_init_time(set_type=set_type),
        test_last_time=dataset.get_last_time(set_type=set_type)
    )

    x1 = pm.get_mean_displacement(times_list=t, nodes=torch.as_tensor([0])).squeeze(0).squeeze(0)
    x2 = pm.get_mean_displacement(times_list=t, nodes=torch.as_tensor([1])).squeeze(0).squeeze(0)

    pm_dist = torch.dot(x1-x2, x1-x2)

    lm_dist = lm.get_pairwise_distances(
        times_list=t, node_pairs=torch.as_tensor([[0], [1]])
    )

    # print( pm._x_init )

    print(pm_dist, lm_dist)