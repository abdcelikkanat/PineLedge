import os
import sys
import torch
sys.path.insert(0, '.')
sys.path.insert(1, '..')
import utils
import pickle
from src.experiments import Experiments
from src.learning import LearningModel
from src.estimation import Estimation
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from src.events import Events
import numpy as np

# Parameter Definitions #
datasetname = "resistance_game=4" #"4nodedense" #"resistance_game=4" #"ia_enron" #"resistance_game=4"
dim = 2
bins_num = 10
learning_rate = 0.1
epochs_num = 150 #100  # 500
seed = 666661

batch_size = 9  # resistance_game=4 = 9  #151  # Lyon=242 # # ia_enron=151 # ia_contacts=113
steps_per_epoch = 5
shuffle = True
verbose = True

time_normalization = 1
init_time = 0.0
last_time = 1.0
split_time = 0.8
#########################

# Experiments
samples_bins_num = 20
filename = f"{datasetname}_tInit={init_time}_tLast={last_time}_sTime={split_time}_binNum={samples_bins_num}_subsampling=0"
modelname = f"{datasetname}_scaled={time_normalization}_split={split_time}_D={dim}_B={bins_num}_lr={learning_rate}_epoch={epochs_num}_batch_size={batch_size}_batch_steps={steps_per_epoch}_seed={seed}"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', "datasets", "real", datasetname, f"{datasetname}_events.pkl")
# dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', "datasets", "synthetic", f"{datasetname}_events.pkl")


# Read the test samples
sample_dict_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../experiments/samples/{filename}_dict.pkl")
with open(sample_dict_file_path, 'rb') as f:
    sample_file = pickle.load(f)
test_samples, test_labels = sample_file["test"]["samples"], sample_file["test"]["labels"]

# Initialize the Events
events = Events(seed=seed, batch_size=batch_size)
# Read the events
events.read(dataset_path)
# Normalize the timeline to [0, 1]
if time_normalization:
    events.normalize(init_time=init_time, last_time=last_time)
else:
    init_time = events.get_first_event_time()
    last_time = events.get_last_event_time()
# Get the training and testing events
train_events = events.get_train_events(last_time=split_time, batch_size=batch_size)
# Get the number of nodes in the whole dataset
nodes_num = events.number_of_nodes()
print(f"Number of nodes in the events set: {nodes_num}")
print(f"Number of events in the training set: {train_events.number_of_events()}")
print(f"Number of events in the testing set: {train_events.number_of_events()}")


# Load the dataset
data_loader = DataLoader(train_events, batch_size=1, shuffle=shuffle, collate_fn=utils.collate_fn)

with open("../experiments/outputs/prior_weight_analysis.txt", 'a') as file:
    file.write(f"Dataset: {datasetname}\n")
    prior_weights = [0., 0.0001, 0.001, 0.01, 0.1, 1.0] #[-1.0] #
    for pw in prior_weights:
        file.write(f"Weight: {pw}\n")
        # Model path
        model_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       f"../experiments/models/prior_analysis_{modelname}_pw={pw}.model")
        # Get the model
        lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=split_time,
                           learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch, verbose=verbose,
                           seed=seed, prior_weight=pw)
        lm.load_state_dict(torch.load(model_file_path))
        est = Estimation(lm=lm, init_time=init_time, split_time=split_time, last_time=last_time)


        sample_dict_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../experiments/samples/{filename}_dict.pkl")
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


        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        file.write(f"Train AUC: {auc}\n")

        # s = est.get_mean_vt(times_list=torch.as_tensor([0.9, 0.95]), nodes=torch.as_tensor([0, 1, 2]))
        # print(s)

        test_samples, test_labels = sample_file["test"]["samples"], sample_file["test"]["labels"]
        labels = test_labels
        samples = test_samples
        y_pred = []
        y_true = []
        for i, j in zip(*np.triu_indices(nodes_num, k=1)):

            if len(labels[i][j]):

                # print(samples[i][j])
                y_pred_ij = est.get_log_intensity(time_list=samples[i][j], node_pairs=[[i], [j]]).tolist()
                y_pred.extend(y_pred_ij)

                y_true.extend(labels[i][j])
                # assert len(y_true) == len(y_pred), f"OLMAZ {len(y_true)} == {len(y_pred)}. {y_pred_ij}"

        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        file.write(f"Test AUC: {auc}\n")

