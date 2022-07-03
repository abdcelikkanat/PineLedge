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

from argparse import ArgumentParser, RawTextHelpFormatter


########################################################################################################################
parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--input', type=str, required=True, help='Path of the dataset'
)
parser.add_argument(
    '--model_path', type=str, required=True, help='Path of the model'
)
parser.add_argument(
    '--samples_path', type=str, required=True, help='Path of the samples'
)
parser.add_argument(
    '--output_path', type=str, required=True, help='Path of the output file'
)

parser.add_argument(
    '--bins_num', type=int, default=100, required=False, help='Number of bins'
)
parser.add_argument(
    '--dim', type=int, default=2, required=False, help='Dimension size'
)
parser.add_argument(
    '--k', type=int, default=10, required=False, help='Latent dimension size of the prior element'
)
parser.add_argument(
    '--prior_lambda', type=float, default=1e5, required=False, help='Scaling coefficient of the covariance'
)
parser.add_argument(
    '--epochs_num', type=int, default=100, required=False, help='Number of epochs'
)
parser.add_argument(
    '--spe', type=int, default=1, required=False, help='Number of steps per epoch'
)
parser.add_argument(
    '--lr', type=float, default=0.1, required=False, help='Learning rate'
)
parser.add_argument(
    '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
)
parser.add_argument(
    '--verbose', type=bool, default=0, required=False, help='Verbose'
)

args = parser.parse_args()
########################################################################################################################

# # Set some parameters
dataset_path = args.input
model_path = args.model_path
samples_path = args.samples_path
output_path = args.output_path

bins_num = args.bins_num
dim = args.dim
K = args.k
prior_lambda = args.prior_lambda
epochs_num = args.epochs_num
steps_per_epoch = args.spe
learning_rate = args.lr

seed = args.seed
verbose = args.verbose

########################################################################################################################

# Load the dataset
all_events = Events(seed=seed)
all_events.read(dataset_path)
batch_size = all_events.number_of_nodes()

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
lm.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

########################################################################################################################

test_intervals_num = 8

# Load sample files
with open(samples_path, 'rb') as f:
    samples_data = pkl.load(f)
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
params = lm.get_hyperparameters()
print(params.keys())
########################################################################################################################
file = open(output_path, 'w')

# Predicted scores
pred_scores = [[] for _ in range(test_intervals_num)]
for b in range(test_intervals_num):
    for sample in samples[b]:
        i, j, t_init, t_last = sample
        #score = lm.get_intensity_integral_for(i=i, j=j, interval=torch.as_tensor([t_init, t_last]))
        score = lm.get_log_intensity(times_list=torch.as_tensor([0]), node_pairs=torch.as_tensor([[i], [j]]))
        # print(t_init, t_last, score, torch.exp(params['_beta'][i] + params['_beta'][j] - ((params['_x0'][i] - params['_x0'][j])**2).sum()))
        pred_scores[b].append(score)


for b in range(test_intervals_num):

    if sum(labels[b]) != len(labels[b]) and sum(labels[b]) != 0:  

        roc_auc = roc_auc_score(y_true=labels[b], y_score=pred_scores[b])
        file.write(f"Roc AUC, Bin Id {b}: {roc_auc}\n")
        pr_auc = average_precision_score(y_true=labels[b], y_score=pred_scores[b])
        file.write(f"PR AUC, Bin Id {b}: {pr_auc}\n")
        file.write("\n")

roc_auc_complete = roc_auc_score(
    y_true=[l for bin_labels in labels for l in bin_labels],
    y_score=[s for bin_scores in pred_scores for s in bin_scores]
)
file.write(f"Roc AUC in total: {roc_auc_complete}\n")
pr_auc_complete = average_precision_score(
    y_true=[l for bin_labels in labels for l in bin_labels],
    y_score=[s for bin_scores in pred_scores for s in bin_scores]
)
file.write(f"PR AUC in total: {pr_auc_complete}\n")

file.close()
