import os
import utils
import torch
from src.learning import LearningModel
from src.prediction import PredictionModel
from src.events import Events
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle as pkl
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import roc_auc_score

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

##################################################################################################################
#  Link reconstruction experiment  #
##################################################################################################################
# Dataset name
dataset_name = f"ia-contacts_hypertext2009"
# Percentage
p = 0.1

# Define dataset
dataset_folder = os.path.join(utils.BASE_FOLDER, "experiments", "samples_reconst", f"{dataset_name}_p={p}_normalized")

# Load the dataset
residual_events = Events(path=dataset_folder, seed=seed)
# Print dataset info
residual_events.info()
# # Normalize the events # We don't need it
# all_events.normalize(init_time=0, last_time=1.0)

# Load sample files
with open(os.path.join(dataset_folder, "samples.pkl"), 'rb') as f:
    samples_data = pkl.load(f)
    pos_samples, neg_samples = samples_data["pos"], samples_data["neg"]

# Let's run the model
nodes_num = residual_events.number_of_nodes()
# Run the model
dim = 2
K = 4
bins_num = 3
prior_lambda = 1e5
batch_size = nodes_num  #1
learning_rate = 0.01
epochs_num = 240  # 500
steps_per_epoch = 3
seed = utils.str2int("testing_reconstruction")
verbose = True
shuffle = True
learn = True

# Construct the data for the learning model
residual_data = residual_events.get_pairs(), residual_events.get_events()
# Initiate the constructor
lm = LearningModel(
    data=residual_data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,  last_time=1., batch_size=batch_size,
    prior_k=K, prior_lambda=prior_lambda,
    learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
    verbose=verbose, seed=seed
)
# Learn the model
model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pl={prior_lambda}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}"
model_path = os.path.join(utils.BASE_FOLDER, "experiments", "models", f"{model_name}.model")
if learn:
    lm.learn()
    torch.save(lm.state_dict(), model_path)
else:
    lm.load_state_dict(torch.load(model_path))

# Returns a list storing intensity integrals of an upper triangular matrix
intensity_values = lm.get_intensity_integral(nodes=torch.arange(nodes_num, dtype=torch.int))

# Combine the samples
samples = pos_samples + neg_samples
# Combine the labels
true_labels = [1] * len(pos_samples) + [0] * len(neg_samples)
# Compute the prediction scores
pred_scores = [intensity_values[utils.pairIdx2flatIdx(i=pair[0], j=pair[1], n=nodes_num)] for pair in samples]

# Compute some metrics
nmis = normalized_mutual_info_score(labels_true=true_labels, labels_pred=pred_scores)
amis = adjusted_mutual_info_score(labels_true=true_labels, labels_pred=pred_scores)
auc = roc_auc_score(y_true=true_labels, y_score=pred_scores)
print(f"Normalized Mutual Info Score: {nmis}")
print(f"Adjusted Mutual Info Score: {amis}")
print(f"ROC AUC Score: {auc}")



##################################################################################################################
#  Future link prediction experiment  #
##################################################################################################################
# Dataset name
dataset_name = f"ia-contacts_hypertext2009"
# Split time point
split_time = 0.9

# Define dataset
dataset_folder = os.path.join(utils.BASE_FOLDER, "experiments", "samples_linkpred", f"{dataset_name}_st={split_time}_normalized")

# Load the dataset
residual_events = Events(path=dataset_folder, seed=seed)
# Print dataset info
residual_events.info()
# # Normalize the events # We don't need it
# all_events.normalize(init_time=0, last_time=1.0)

# Load sample files
with open(os.path.join(dataset_folder, "samples.pkl"), 'rb') as f:
    samples_data = pkl.load(f)
    pos_samples, neg_samples = samples_data["pos"], samples_data["neg"]

# Let's run the model
nodes_num = residual_events.number_of_nodes()
# Run the model
dim = 2
K = 4
bins_num = 3
prior_lambda = 1e5
batch_size = nodes_num  #1
learning_rate = 0.01
epochs_num = 240  # 500
steps_per_epoch = 1
seed = utils.str2int("testing_prediction")
verbose = True
shuffle = True
learn = True

# Construct the data for the learning model
residual_data = residual_events.get_pairs(), residual_events.get_events()
# Initiate the constructor
lm = LearningModel(
    data=residual_data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,  last_time=1., batch_size=batch_size,
    prior_k=K, prior_lambda=prior_lambda,
    learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
    verbose=verbose, seed=seed
)
# Learn the model
model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pl={prior_lambda}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}"
model_path = os.path.join(utils.BASE_FOLDER, "experiments", "models", f"{model_name}.model")
if learn:
    lm.learn()
    torch.save(lm.state_dict(), model_path)
else:
    lm.load_state_dict(torch.load(model_path))

# Define the prediction model
pm = PredictionModel(lm=lm, pred_init_time=split_time, pred_last_time=1.0)

# Returns a list storing intensity integrals of an upper triangular matrix
intensity_values = pm.get_intensity_integral(nodes=torch.arange(nodes_num, dtype=torch.int))

# Combine the samples
samples = pos_samples + neg_samples
# Combine the labels
true_labels = [1] * len(pos_samples) + [0] * len(neg_samples)
# Compute the prediction scores
pred_scores = [intensity_values[utils.pairIdx2flatIdx(i=pair[0], j=pair[1], n=nodes_num)] for pair in samples]

# Compute some metrics
nmis = normalized_mutual_info_score(labels_true=true_labels, labels_pred=pred_scores)
amis = adjusted_mutual_info_score(labels_true=true_labels, labels_pred=pred_scores)
auc = roc_auc_score(y_true=true_labels, y_score=pred_scores)
print(f"Normalized Mutual Info Score: {nmis}")
print(f"Adjusted Mutual Info Score: {amis}")
print(f"ROC AUC Score: {auc}")
