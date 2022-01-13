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

# Dataset name
dataset_name = "resistance_game=4" #"LyonSchool" #"resistance_game=4"

# Define the dataset path
dataset_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "datasets", "real", dataset_name, f"{dataset_name}_events.pkl"
)

# Set some parameters
dim = 2
bins_num = 10
batch_size = 4
learning_rate = 0.01
epochs_num = 300
seed = 123
verbose = True
time_normalization = True
shuffle = True
actions = ["learn", "animate"] # plot_events

# Define the model path
model_file_path = os.path.join(
    "../outputs/", f"{dataset_name}_lr={learning_rate}_batch={batch_size}_B={bins_num}_D={dim}_scale={time_normalization}_seed={seed}.model"
)

# Load the dataset
dataset = Dataset(dataset_path, init_time=0, time_normalization=time_normalization, train_ratio=1.0)
print(f"Number of nodes: {dataset.get_nodes_num()}")

# Run the model
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=utils.collate_fn)
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
                     figure_path=f"../outputs/{dataset_name}_gauss+periodic.html"
                     )

# Experiments
if "expereval" in actions:
    from sklearn.metrics import roc_auc_score

    # Load the model
    lm.load_state_dict(torch.load(model_file_path))

    exp = Experiments(dataset=dataset, set_type="test")
    samples, labels = exp.get_samples(), exp.get_labels()
    # exp.plot_events(u=2, v=3, bins=100)

    # s, l = [], []
    # for x, y in zip(samples, labels):
    #     if x[0] == 0 and x[1] == 1:
    #        s.append(x)
    #        l.append(y)
    # samples, labels = s, l
    # exp.get_sample_stats()
    print("Samples constructed!")
    F = exp.get_freq_map(set_type="train")
    print("Frequency map is ok!")
    simon_pred = [F[sample[0], sample[1]] for sample in samples]
    simon_auc = roc_auc_score(y_true=labels, y_score=simon_pred)
    simon_aps = average_precision_score(y_true=labels, y_score=simon_pred)
    print(f"Simon auc: {simon_auc} / aps: {simon_aps}")

    pm = PredictionModel(
        lm=lm, test_init_time=dataset.get_init_time(set_type="test"),
        test_last_time=dataset.get_last_time(set_type="test")
    )

    # mean_vt = pm.get_mean_vt(times_list=torch.as_tensor([0.9, 0.95, 0.98]))
    # mean_xt = pm.get_mean_xt(times_list=torch.as_tensor([0.9, 0.95, 0.98]), nodes=torch.as_tensor([0,1]))
    # t = pm.get_log_likelihood(times_list=torch.as_tensor([0.9, 0.95, 0.98]), nodes=torch.as_tensor([0, 1]))
    # t = pm.get_log_intensity(times_list=torch.as_tensor([0.9, 0.95, 0.98]), node_pair=torch.as_tensor([[1], [2]]))
    # t = pm.get_negative_log_likelihood(times_list=torch.as_tensor([0.9, 0.95, 0.98]), node_pair=torch.as_tensor([[1], [2]]))
    # print(t)

    distances = []
    for sample in samples:

        score = pm.get_negative_log_likelihood(
            times_list=torch.as_tensor([sample[2], ], dtype=torch.float), node_pair=torch.as_tensor([[sample[0]], [sample[1]]])
        )
        distances.append(score)

    # model_params = lm.get_model_params()
    # distances = []
    #
    # for sample in samples:
    #
    #     score = lm.prediction(
    #         event_times=torch.as_tensor([sample[2]]),
    #         event_node_pairs=torch.as_tensor([[sample[0], ], [sample[1], ]]), test_middle_point=torch.as_tensor([9.5])
    #     )
    #     # print(score)
    #     distances.append(score)
    #
    #     # distances.append(
    #     #     -lm.get_log_intensity(times_list=torch.as_tensor([sample[2]]), node_pairs=torch.as_tensor([sample[0:2]]).t())
    #     #     +lm.get_intensity_integral(node_pairs=torch.as_tensor([sample[0:2]]).t())
    #     # )
    #
    #
    pred_auc = roc_auc_score(y_true=labels, y_score=distances)
    pred_aps = average_precision_score(y_true=labels, y_score=distances)
    print(f"pred auc: {pred_auc} / aps: {pred_aps}")