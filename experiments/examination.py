import torch
from src.base import BaseModel
from src.learning import LearningModel
from torch.utils.data import DataLoader
from src.events import Events
import numpy as np
import pandas as pd
from visualization.animation import Animation
import os
import sys
import utils
import pickle as pkl
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

########################################################################################################################

# Set some paremeters
dim = 2
K = 10
bins_num = 32
prior_lambda = 1e5
#batch_size = 2  #1
learning_rate = 0.1
epochs_num = 100  # 500
steps_per_epoch = 1
seed = utils.str2int("full2")
verbose = True
shuffle = True

suffix = "" #f"_percent={0.1}" #f"_percent={0.01}" #f"_percent={0.2}" #"_nhpp" #"_survival"
###
dataset_name = "ia-contacts_hypertext2009" #f"two_clusters_fp_sizes=10_10_beta=2" #f"four_nodes_fp" #f"two_clusters_fp_sizes=10_10_beta=1" #f"three_clusters_sizes=15_20_10" # sbm_survival three_clusters_fp_sizes=15_20_10_beta=0
model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pl={prior_lambda}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}{suffix}"

# Define dataset and model path
dataset_folder = os.path.join(
    utils.BASE_FOLDER, "datasets", "real", dataset_name #synthetic
)

# Define dataset and model path
#dataset_folder = os.path.join(utils.BASE_FOLDER, "datasets", "synthetic", dataset_name)
dataset_folder = os.path.join(utils.BASE_FOLDER, "datasets", "real", dataset_name)
samples_file_path=os.path.join(f"/work3/abce/workspace/pivem/samplesv{test_intervals_num}/", dataset_name)
#samples_file_path=os.path.join(utils.BASE_FOLDER, f"baselines/samplesv{test_intervals_num}", dataset_name)

########################################################################################################################

# Load the dataset
all_events = Events(seed=seed)
all_events.read(dataset_folder)

batch_size=all_events.number_of_nodes()
model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pl={prior_lambda}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}{suffix}"
model_path = os.path.join(os.path.join(utils.BASE_FOLDER, "experiments", "models", model_name), f"{model_name}.model")


# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)

# Get the number of nodes
nodes_num = all_events.number_of_nodes()

data = all_events.get_pairs(), all_events.get_events()

# the model
lm = LearningModel(data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=1., batch_size=batch_size,
                   prior_k=K, prior_lambda=prior_lambda,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                   verbose=verbose, seed=seed)

# Load the model
assert os.path.exists(model_path), "The model file does not exist!"
lm.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

########################################################################################################################

with open(os.path.join(samples_file_path, "samples.pkl"), 'rb') as f:
    samples_data = pkl.load(f)
    #valid_labels = samples_data["valid_labels"]
    #valid_samples = samples_data["valid_samples"]
    labels = samples_data["test_labels"]
    samples = samples_data["test_samples"]

'''
# Construct samples

threshold = 1
test_intervals_num = 8
test_intervals = torch.linspace(0, 1.0, test_intervals_num+1)

samples, labels = [[] for _ in range(test_intervals_num)], [[] for _ in range(test_intervals_num)]
for i, j in utils.pair_iter(n=nodes_num):
    pair_events = torch.as_tensor(all_events[(i, j)][1])

    for b in range(test_intervals_num):
        test_bin_event_counts = torch.sum( (test_intervals[b+1] > pair_events) * (pair_events >= test_intervals[b]) )

        samples[b].append([i, j, test_intervals[b], test_intervals[b+1]])

        if test_bin_event_counts >= threshold:
            labels[b].append(1)
        else:
            labels[b].append(0)
'''
########################################################################################################################
# Predicted scores
pred_scores = [[] for _ in range(test_intervals_num)]
for b in range(test_intervals_num):
    for sample in samples[b]:
        i, j, t_init, t_last = sample
        score = lm.get_intensity_integral_for(i=i, j=j, interval=torch.as_tensor([t_init, t_last]))
        pred_scores[b].append(score)


for b in range(test_intervals_num):
    if len(labels[b]) != sum(labels[b]) and sum(labels[b]) != 0:
        roc_auc = roc_auc_score(y_true=labels[b], y_score=pred_scores[b])
        print(f"Roc AUC, Bin Id {b}: {roc_auc}")
        pr_auc = average_precision_score(y_true=labels[b], y_score=pred_scores[b])
        print(f"PR AUC, Bin Id {b}: {pr_auc}")
        print("")

roc_auc_complete = roc_auc_score(
    y_true=[l for bin_labels in labels for l in bin_labels],
    y_score=[s for bin_scores in pred_scores for s in bin_scores]
)
print(f"Roc AUC in total: {roc_auc_complete}")
pr_auc_complete = average_precision_score(
    y_true=[l for bin_labels in labels for l in bin_labels],
    y_score=[s for bin_scores in pred_scores for s in bin_scores]
)
print(f"PR AUC in total: {pr_auc_complete}")

'''
nodes_num = all_events.number_of_nodes()
intensities = lm.get_intensity_integral(nodes=torch.arange(nodes_num))

M = torch.zeros(size=(nodes_num, nodes_num))
for i, j in utils.pair_iter(n=nodes_num):
    M[i, j] = intensities[utils.pairIdx2flatIdx(i=i, j=j, n=nodes_num)]

plt.figure()
plt.matshow(M)
plt.colorbar()
plt.show()
'''
