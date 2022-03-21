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
from src.animation import Animation

#########################
# Define the parameters #
dim = 2
bins_num = 10
learning_rate = 0.1
epochs_num = 300  # 500
seed = 999990

batch_size = 242  # Lyon=242
steps_per_epoch = 5
shuffle = True
verbose = True

time_normalization = 1
init_time = 0.0
last_time = 1.0
split_time = 0.1

datasetname = "LyonSchool" # resistance_game=4
samples_bins_num = 20
#########################


filename = f"{datasetname}_tInit={init_time}_tLast={last_time}_sTime={split_time}_binNum={samples_bins_num}_subsampling=0"
modelname = f"{datasetname}_scaled={time_normalization}_split={split_time}_D={dim}_B={bins_num}_lr={learning_rate}_epoch={epochs_num}_batch_size={batch_size}_batch_steps={steps_per_epoch}_seed={seed}"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', "datasets", "real", datasetname, f"{datasetname}_events.pkl")
model_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../experiments/models/{modelname}.model")

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

print(init_time, last_time, split_time)
# Get the training events
train_events = events.get_train_events(last_time=split_time, batch_size=batch_size)
# Get the number of nodes in the whole dataset
nodes_num = events.number_of_nodes()

# Load the dataset
data_loader = DataLoader(train_events, batch_size=1, shuffle=shuffle, collate_fn=utils.collate_fn)
# Get the model
lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=split_time,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch, verbose=verbose,
                   seed=seed)
lm.load_state_dict(torch.load(model_file_path))

times_list = torch.linspace(0, 0.2, 100)
node_ids = np.tile(np.arange(nodes_num).reshape(1, -1), (len(times_list), 1)).flatten()
node_times = np.tile(np.asarray(times_list).reshape(-1, 1), (1, nodes_num)).flatten()
colors = ['r', 'b', 'g', 'm']


embs_pred = lm.get_xt(times_list=times_list).detach().numpy()
anim = Animation(embs=embs_pred, time_list=node_times, group_labels=node_ids,
                 colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
                 title="Prediction model"+" ".join(filename.split('_')),
                 padding=0.0,
                 dataset=None,
                 # figure_path=f"../outputs/4nodes.html"
                 )

