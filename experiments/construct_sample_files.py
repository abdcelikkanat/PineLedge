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
# subsampling
subsampling = False
# Set bins_num
bins_num = 1


# Load the dataset
all_events = Events(path=dataset_folder, seed=seed)
# Print dataset info
all_events.info()
# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)


'''
# ##################################
# # Reconst whole network experiment #
# ##################################
#   Let's remove some percent of links at random and predict them

all_labels, all_samples = all_events.construct_samples(
    bins_num=bins_num, init_time=0, last_time=1.0, with_time=False, parent_events=all_events, subsampling=subsampling
)

test_weights = []
for label, sample in zip(all_labels, all_samples):

    pair = (sample[0], sample[1])
    sample_events = [e for e in all_events[pair][1] if sample[3] > e >= sample[2]]

    test_weights.append(len(sample_events))

sample_folder = os.path.join(utils.BASE_FOLDER, "xexperiments", f"samples_full_reconst_bins={bins_num}", dataset_name + f"_normalized")
# Create folder
if not os.path.exists(sample_folder):
    os.makedirs(sample_folder)

# Write the events
all_events.write(folder_path=sample_folder)

# Write the combined links
with open(os.path.join(sample_folder, f"{dataset_name}_residual_combined.edges"), 'w') as f:
    for sample in all_events.get_pairs():
        f.write(f"{sample[0]} {sample[1]}\n")

# Write the combined links
with open(os.path.join(sample_folder, f"{dataset_name}_residual_combined_weighted.edges"), 'w') as f:
    for sample in all_events.get_pairs():
        f.write(f"{sample[0]} {sample[1]} {len(all_events[(sample[0], sample[1])][1])}\n")

# Write the samples
with open(os.path.join(sample_folder, "samples.pkl"), 'wb') as f:
    pkl.dump({"test_labels":  all_labels, "test_samples": all_samples, "test_weights": test_weights}, f)

'''



# ##################################
# # Link reconstruction experiment #
# ##################################
#   Let's remove some percent of links at random and predict them
for p in [0.05]:  #[0.05, 0.1, 0.2]:

    num = int(all_events.number_of_event_pairs() * p * 2)
    temp_events, valid_events = all_events.remove_events(num=num)
    train_events, test_events = temp_events.remove_events(num=num)

    # valid_events, test_events = removed_events.remove_events(num=removed_events.number_of_event_pairs() // 2)

    valid_labels, valid_samples = valid_events.construct_samples(
        bins_num=bins_num, init_time=0, last_time=1.0, with_time=False, parent_events=all_events, subsampling=subsampling
    )

    test_labels, test_samples = test_events.construct_samples(
        bins_num=bins_num, init_time=0, last_time=1.0, with_time=False, parent_events=all_events, subsampling=subsampling
    )


    valid_weights = []
    for label, sample in zip(valid_labels, valid_samples):

        pair = (sample[0], sample[1])
        sample_events = [e for e in all_events[pair][1] if sample[3] > e >= sample[2]]

        valid_weights.append(len(sample_events))

    test_weights = []
    for label, sample in zip(test_labels, test_samples):

        pair = (sample[0], sample[1])
        sample_events = [e for e in all_events[pair][1] if sample[3] > e >= sample[2]]

        test_weights.append(len(sample_events))

    sample_folder = os.path.join(utils.BASE_FOLDER, "experiments", f"samples_reconst_bins={bins_num}", dataset_name + f"_p={p}_normalized")
    # Create folder
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Write the events
    train_events.write(folder_path=sample_folder)

    # Write the combined links
    with open(os.path.join(sample_folder, f"{dataset_name}_residual_combined.edges"), 'w') as f:
        for sample in train_events.get_pairs():
            f.write(f"{sample[0]} {sample[1]}\n")

    # Write the combined links
    with open(os.path.join(sample_folder, f"{dataset_name}_residual_combined_weighted.edges"), 'w') as f:
        for sample in train_events.get_pairs():
            f.write(f"{sample[0]} {sample[1]} {len(train_events[(sample[0], sample[1])][1])}\n")

    # Write the samples
    with open(os.path.join(sample_folder, "samples.pkl"), 'wb') as f:
        pkl.dump({"valid_labels": valid_labels, "valid_samples": valid_samples, "valid_weights": valid_weights,
                  "test_labels":  test_labels, "test_samples": test_samples, "test_weights": test_weights}, f)


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
