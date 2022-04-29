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
prior_weight_list = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
batch_size = 45  #1
learning_rate = 0.01
epochs_num = 300  # 500
steps_per_epoch = 5
seed = utils.str2int("testing_dyad_removal3")
verbose = False
shuffle = True
####
learn = True
###

dataset_name = f"three_clusters_fp_sizes=15_20_10"

# Define dataset and model path
dataset_path = os.path.join(
    utils.BASE_FOLDER, "datasets", "synthetic", dataset_name, f"{dataset_name}_events.pkl"
)

# Load the dataset
all_events = Events(seed=seed)
all_events.read(dataset_path)

# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)

# Get the number of nodes
nodes_num = all_events.number_of_nodes()

percent = 0.2
num = int(all_events.number_of_event_pairs() * percent)
residual, removed = all_events.remove_events(num=num)

removed_pairs = removed.get_pairs()
removed_events = removed.get_events()

for pw in prior_weight_list:

    ###
    model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pw={pw}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}_percent={percent}"

    model_folder = os.path.join(
        utils.BASE_FOLDER, "experiments", "models", model_name
    )
    model_path = os.path.join(
        model_folder, f"{model_name}.model"
    )

    # Construct data
    data = all_events.get_pairs(), all_events.get_events()
    # Run the model
    lm = LearningModel(data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim, k=K,
                       node_pairs_mask=torch.as_tensor(removed_pairs).T, last_time=1.,
                       learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                       verbose=verbose, seed=seed, pw=pw)

    if learn:
        # Save the model
        os.makedirs(model_folder)

        lm.learn()
        torch.save(lm.state_dict(), model_path)
    else:
        lm.load_state_dict(torch.load(model_path))

    nll = 0
    for node_pair, pair_events in zip(removed_pairs, removed_events):
        nll += lm.get_negative_log_likelihood(
            nodes=torch.as_tensor(node_pair),
            event_times=torch.as_tensor(pair_events),
            event_node_pairs=torch.repeat_interleave(torch.as_tensor(node_pair).unsqueeze(1), len(pair_events), 1)
        )
    print(f"Loss of masked pairs: {nll} - pw: {pw}")


# residual2, removed2 = residual.remove_events(num=num)
# nll = 0
# for node_pair, pair_events in zip(removed2.get_pairs(), removed2.get_events()):
#     nll += lm.get_negative_log_likelihood(
#         nodes=torch.as_tensor(node_pair),
#         event_times=torch.as_tensor(pair_events),
#         event_node_pairs=torch.repeat_interleave(torch.as_tensor(node_pair).unsqueeze(1), len(pair_events), 1)
#     )
#
# print(f"Loss of unmasked pairs: {nll}")