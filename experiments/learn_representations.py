import os
import utils
import torch
from src.learning import LearningModel
from torch.utils.data import DataLoader
from src.events import Events

# global control for device
CUDA = True
# availability for different devices
avail_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# choosing device and setting default tensor, meaning that each new tensor has a default device pointing to
if (CUDA) and (avail_device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


# Set some parameters
dim = 2
K = 3
bins_num = 100
prior_lambda = 1e2 #1e5
#batch_size = 2  #1
learning_rate = 0.1
epochs_num = 500  # 500
steps_per_epoch = 1
seed = utils.str2int("25clusters")
verbose = True
shuffle = True

###
# dataset_name = f"25_clusters_mg_B=100_noise-sigma=0.1_x0-c=2.0_rbf-sigma=0.0001_lambda=1.0_sizes=5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_beta=1.25"
dataset_name = f"four_nodes_fp_beta=1.5"
# dataset_name = f"fixed_two_clusters_fp_sizes=10_10_beta=2"
# dataset_name = f"25_clusters_mg_B=100_noise-sigma=0.1_x0-c=2.0_rbf-sigma=0.0001_lambda=1.0_sizes=5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_5_beta=1.25"
# dataset_name = f"three_clusters_fp_sizes=15_20_10_beta=1"
# dataset_name = f"four_nodes_fp_beta=1.5"
# dataset_name = f"fixed_two_clusters_fp_sizes=10_10_beta=2"
# dataset_name = f"three_clusters_fp_sizes=15_20_10_beta=1"
# dataset_name = f"four_nodes_fp" #f"three_clusters_fp_sizes=15_20_10_beta=1"
# dataset_name = f"four_nodes_fp"
# dataset_name = f"3_clusters_mg_B=100_noise_s=0.1_rbf-s=-9.210290371559083_lambda=1.0_sizes=20_20_20_beta=1.5" #"lyonschool"
# dataset_name = "ia-contacts_hypertext2009"
# dataset_name = f"two_clusters_fp_sizes=10_10_beta=2" #"four_nodes_fp" #f"two_clusters_fp_sizes=10_10_beta=1" #f"fb_forum" #f"three_clusters_fp_sizes=15_20_10"
model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pl={prior_lambda}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}"

# Define dataset and model path
'''
dataset_path = os.path.join(utils.BASE_FOLDER, "datasets", "synthetic", dataset_name)  # synthetic # real)
model_folder = os.path.join(utils.BASE_FOLDER, "experiments", "models", model_name)
model_path = os.path.join(model_folder, f"{model_name}.model")
'''
dataset_path = os.path.join("/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/", dataset_name)
model_folder = os.path.join("/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/models", model_name)
model_path = os.path.join(model_folder, f"{model_name}.model")
log_path = os.path.join("/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/logs", f"{model_name}.txt")

# Load the dataset
all_events = Events(seed=seed)
all_events.read(dataset_path)
batch_size = all_events.number_of_nodes()

# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)

# Get the number of nodes
nodes_num = all_events.number_of_nodes()
data = all_events.get_pairs(), all_events.get_events()

# Run the model
lm = LearningModel(data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,  last_time=1., batch_size=batch_size,
                   prior_k=K, prior_lambda=prior_lambda,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                   verbose=verbose, seed=seed, device=torch.device(avail_device))


# assert not os.path.exists(model_path), "The file exists!"

# Save the model
os.makedirs(model_folder)

lm.learn(loss_file_path=log_path)
torch.save(lm.state_dict(), model_path)
