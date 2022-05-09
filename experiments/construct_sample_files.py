import os
import utils
import torch
from src.learning import LearningModel
from src.events import Events
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle as pkl

# global control for device
CUDA = True
# availability for different devices
avail_device="cuda:0" if torch.cuda.is_available() else "cpu"

# choosing device and setting default tensor, meaning that each new tensor has a default device pointing to
if (CUDA) and (avail_device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Set some parameters
seed = utils.str2int("experiment_design_example")

# Dataset name
dataset_name = f"ia-contacts_hypertext2009" #f"soc-wiki-elec_new" #f"soc-sign-bitcoinalpha_new" #f"soc-wiki-elec_new" #f"ia-contact" #f"fb-forum" #f"ia-contacts_hypertext2009"
# Define dataset
dataset_folder = os.path.join(utils.BASE_FOLDER, "datasets", "real", dataset_name)

# Load the dataset
all_events = Events(path=dataset_folder, seed=seed)
# Print dataset info
all_events.info()
# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)

# ##################################
# # Link reconstruction experiment #
# ##################################
#   Let's remove some percent of links at random and predict them
for p in [0.05, 0.1, 0.2]:
    residual_events, test_pos_samples, test_neg_samples = all_events.remove_events(
        num=int(all_events.number_of_event_pairs() * p )
    )
    residual_events.info()

    sample_folder = os.path.join(utils.BASE_FOLDER, "experiments", "samples_reconst", dataset_name + f"_p={p}_normalized")
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    residual_events.write(folder_path=sample_folder)
    with open(os.path.join(sample_folder, "samples.pkl"), 'wb') as f:
        pkl.dump({"pos": test_pos_samples, "neg": test_neg_samples}, f)


# # ############################
# # Link prediction experiment #
# # ############################
# for st in [0.8, 0.9, 0.95]:
#     residual_events, test_pos_samples, test_neg_samples = all_events.split_events_in_time(
#         split_time=st
#     )
#     residual_events.info()
#
#     sample_folder = os.path.join(utils.BASE_FOLDER, "samples_linkpred", dataset_name + f"_st={st}_normalized")
#     if not os.path.exists(sample_folder):
#         os.makedirs(sample_folder)
#
#     residual_events.write(folder_path=sample_folder)
#     with open(os.path.join(sample_folder, "samples.pkl"), 'wb') as f:
#         pkl.dump({"pos": test_pos_samples, "neg": test_neg_samples}, f)
