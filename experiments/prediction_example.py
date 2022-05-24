import os
import utils
import torch
from src.learning import LearningModel
from torch.utils.data import DataLoader
from src.events import Events
import pickle as pkl
from visualization.animation import Animation
from src.prediction import PredictionModel

# Set some paremeters
dim = 2
K = 4
bins_num = 3
prior_lambda = 1e5
batch_size = 30  #1
learning_rate = 0.01
epochs_num = 800  # 500
steps_per_epoch = 3
seed = utils.str2int("testing_prediction")
verbose = True
split_time = 0.9

learn = False

###
dataset_name = f"three_clusters_fp_sizes=15_20_10_beta=1"
model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pl={prior_lambda}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}"

# Define dataset and model path
dataset_folder = os.path.join(
    utils.BASE_FOLDER, "datasets", "synthetic", dataset_name
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
all_events.read(dataset_folder)

# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)

# Construct the train and test sets
train_events, test_events = all_events.split_events_in_time(split_time=split_time)
train_data = train_events.get_pairs(), train_events.get_events()

# Get the number of nodes
nodes_num = all_events.number_of_nodes()

# Run the model
lm = LearningModel(data=train_data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,  last_time=split_time,
                   batch_size=batch_size, prior_k=K, prior_lambda=prior_lambda,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                   verbose=verbose, seed=seed)

if learn:
    # Save the model
    if not os.path.exists(model_path):
        os.makedirs(model_folder)

    lm.learn()
    torch.save(lm.state_dict(), model_path)
else:

    lm.load_state_dict(torch.load(model_path))


# Define the prediction model
pm = PredictionModel(lm=lm, pred_init_time=split_time, pred_last_time=1.0)
v = pm.get_expected_vt(times_list=torch.as_tensor([0.94]))

# Flatten the test events and pairs
flat_test_events = torch.as_tensor(
    [e for pair_events in test_events.get_events() for e in pair_events], dtype=torch.float
)

flat_test_pairs = torch.repeat_interleave(
    torch.as_tensor(test_events.get_pairs()),
    repeats=torch.as_tensor(list(map(len, test_events.get_events())), dtype=torch.int),
    dim=0
).T

# Compute log-intensities
intensity_list = pm.get_log_intensity(
    times_list=flat_test_events,
    node_pairs=flat_test_pairs
)
print(intensity_list)
print(pm.get_intensity_integral(nodes=torch.arange(3)))

# Compute negative log-likelihood
nll_list = pm.get_negative_log_likelihood(
    event_times=flat_test_events,
    event_node_pairs=flat_test_pairs
)
# print(nll_list)

print("-----")
s = pm.get_intensity_integral_for_bins(boundaries=torch.as_tensor([0.9, 0.92, 0.98, 1.0]), sample_size_per_bin=torch.as_tensor([5, 4, 7]))

print(s.shape)
print(s)