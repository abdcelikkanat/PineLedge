import os
import utils
import torch
from src.learning import LearningModel
from torch.utils.data import DataLoader
from src.events import Events


# Set some paremeters
dim = 2
K = 4
bins_num = 3
pw = 1e3
batch_size = 45  #1
learning_rate = 0.01
epochs_num = 300  # 500
steps_per_epoch = 5
seed = utils.str2int("testing_seq2")
verbose = True
shuffle = True

###
dataset_name = f"three_clusters_fp_sizes=15_20_10"
model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pw={pw}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}"

# Define dataset and model path
dataset_path = os.path.join(
    utils.BASE_FOLDER, "datasets", "synthetic", dataset_name, f"{dataset_name}_events.pkl"
)
model_folder = os.path.join(
    utils.BASE_FOLDER, "experiments", "models", model_name
)
model_path = os.path.join(
    model_folder, f"{model_name}.model"
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
lm = LearningModel(data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim, k=K, last_time=1.,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                   verbose=verbose, seed=seed, pw=pw)


# assert not os.path.exists(model_path), "The file exists!"

# Save the model
# os.makedirs(model_folder)

lm.learn()
torch.save(lm.state_dict(), model_path)
