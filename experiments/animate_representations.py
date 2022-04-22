import torch
from src.base import BaseModel
from src.learning import LearningModel
from torch.utils.data import DataLoader
from src.events import Events
import numpy as np
from src.estimation import Estimation
import pandas as pd
from visualization.animation import Animation
import os
import sys
import utils
import pickle as pkl


# Set some paremeters
dim = 2
bins_num = 3 #3 #5
pw = 1e-6 #100000 #1.0  #1.0
batch_size = 45 #45 #70 #45  #1
learning_rate = 0.1
epochs_num = 300 #100 #400  # 500
steps_per_epoch = 5
seed = utils.str2int("testing_cv_full")  # all_events # survival_true_with_event # nhpp
verbose = True
shuffle = True
suffix = "" #f"_percent={0.01}" #f"_percent={0.2}" #"_nhpp" #"_survival"

###
dataset_name = "three_clusters_fp_sizes=15_20_10" #f"three_clusters_sizes=15_20_10" # sbm_survival
model_name = f"{dataset_name}_D={dim}_B={bins_num}_pw={pw}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}"+suffix

# Define dataset and model path
dataset_folder = os.path.join(
    utils.BASE_FOLDER, "datasets", "synthetic", dataset_name
)
dataset_path = os.path.join(
    dataset_folder, f"{dataset_name}_events.pkl"
)
model_folder = os.path.join(
    utils.BASE_FOLDER, "experiments", "models", model_name
)
model_path = os.path.join(
    model_folder, f"{model_name}.model"
)
anim_path = os.path.join(
    utils.BASE_FOLDER, "experiments", "animations", f"{model_name}.mp4"
)

# Load the dataset
all_events = Events(seed=seed)
all_events.read(dataset_path)

# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)

# Get the number of nodes
nodes_num = all_events.number_of_nodes()

data = all_events.get_pairs(), all_events.get_events()

# Run the model
lm = LearningModel(data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=1.,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                   verbose=verbose, seed=seed, pw=pw)

# Load the model
assert os.path.exists(model_path), "The model file does not exist!"
lm.load_state_dict(torch.load(model_path))


times_list = torch.linspace(0, 1.0, 100)
# Ground truth animation
# bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_rwidth=bins_rwidth)
# embs_gt = bm.get_xt(times_list=times_list*last_time).detach().numpy()
# node2color = [0] * cluster_sizes[0] + [1] * cluster_sizes[1] + [2] * cluster_sizes[2]
# anim = Animation(embs_gt, fps=12, node2color=node2color)
# anim.save("./three_clusters_gt_sil.mp4")

# Read the group information
node2group_filepath = os.path.join(dataset_folder, f"{dataset_name}_node2group.pkl")
with open(node2group_filepath, "rb") as f:
    node2group_data = pkl.load(f)
node2group, group2node = node2group_data["node2group"], node2group_data["group2node"]

embs_pred = lm.get_xt(times_list=times_list).detach().numpy()
node2color = [node2group[idx] for idx in range(nodes_num)]
anim = Animation(embs_pred, fps=12, node2color=node2color)
anim.save(anim_path)
