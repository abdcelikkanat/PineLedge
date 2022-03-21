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
datasetname = "ia_enron" #"resistance_game=4"
dim = 2
bins_num = 10
learning_rate = 0.1
epochs_num = 100 #300 #1000  # 500
seed = 999990 #30

batch_size = 151 #9
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
model_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../experiments/models/{modelname}.model")

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
# Get the model
lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=split_time,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch, verbose=verbose,
                   seed=seed)
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
        # assert len(y_true) == len(y_pred), f"OLMAZ {len(y_true)} == {len(y_pred)}. {y_pred_ij}"

    else:
        pass #print(i,j)

auc = roc_auc_score(y_true=y_true, y_score=y_pred)
print(f"Train AUC: {auc}")

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
print(f"Test AUC: {auc}")


# if viz == 1:
#     # Visualization
#     times_list = torch.linspace(0, last_time, 100)
#     node_ids = np.tile(np.arange(nodes_num).reshape(1, -1), (len(times_list), 1)).flatten()
#     node_times = np.tile(np.asarray(times_list).reshape(-1, 1), (1, nodes_num)).flatten()
#     colors = ['r', 'b', 'g', 'm']
#     # # Ground truth animation
#     # bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_rwidth=bins_rwidth)
#     # embs_gt = bm.get_xt(times_list=times_list).detach().numpy().reshape(-1, 2)
#     # anim = Animation(embs=embs_gt, time_list=node_times, group_labels=node_ids,
#     #                  colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
#     #                  title="Ground-truth model"+" ".join(filename.split('_')),
#     #                  padding=0.1
#     #                  )
#
#     # Prediction animation
#     # print(times_list)
#     embs_pred = lm.get_xt(times_list=times_list/last_time).detach().numpy()
#     anim = Animation(embs=embs_pred, time_list=node_times, group_labels=node_ids,
#                      colors=[colors[id%len(colors)] for id in node_ids], color_name="Nodes",
#                      title="Prediction model"+" ".join(filename.split('_')),
#                      padding=0.1,
#                      dataset=None,
#                      # figure_path=f"../outputs/4nodes.html"
#                      )
#
# # Experiments
# from sklearn.metrics import roc_auc_score
#
# exp = Experiments(seed=seed)
# exp.set_events(events=train_events)
# labels, samples = exp.construct_samples(bins_num=10, subsampling=0, with_time=True, init_time=init_time, last_time=split_time)
#
# #exp.plot_samples(labels=labels, samples=samples)
#
# F = exp.get_freq_map()
# simon_auc = roc_auc_score(y_true=labels, y_score=[F[sample[0], sample[1]] for sample in samples])
# print(f"Simon auc score: {simon_auc}")
#
# # model_params = lm.get_model_params()
# distances = []
# for sample in samples:
#     # print(sample[0:2])
#     d = lm.get_pairwise_distances(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
#     # distances.append(torch.exp(model_params['beta'][sample[0]] + model_params['beta'][sample[1]] - d))
#     # distances.append(torch.exp( - d))
#     # distances.append(d)
#     # distances.append(
#     #     +lm.get_log_intensity(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
#     #     +lm.get_intensity_integral(node_pairs=torch.as_tensor([sample[0:2]]).t())
#     # )
#     # distances.append(
#     #     +lm.get_log_intensity(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
#     #     -lm.get_intensity_integral(node_pairs=torch.as_tensor([sample[0:2]]).t())
#     # )
#     distances.append(
#         lm.get_log_intensity(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
#     )
#
# pred_auc = roc_auc_score(y_true=labels, y_score=distances)
# print(f"pred auc score: {pred_auc}")