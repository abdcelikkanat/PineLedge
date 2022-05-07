import os
import utils
import torch
from src.learning import LearningModel
from src.events import Events
import matplotlib.pyplot as plt
from sklearn import metrics

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
dataset_name = f"ia-contacts_hypertext2009"
# Define dataset
dataset_folder = os.path.join(utils.BASE_FOLDER, "datasets", "real", dataset_name)

# Load the dataset
all_events = Events(path=dataset_folder, seed=seed)
# Print dataset info
all_events.info()
# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)
# print(all_events[[all_events.number_of_nodes()-2, all_events.number_of_nodes()-1]])
# for i, j in utils.pair_iter(n=all_events.number_of_nodes()):
#     print( len(all_events[(i,j)]) )

###########################
# Dyad removal experiment #
###########################
#   Let's remove 10% of links at random and predict them
residual_events, removed_events = all_events.remove_events(num=int(all_events.number_of_event_pairs() * 0.1))
removed_events.info()

# Let's run the model
nodes_num = all_events.number_of_nodes()
print(nodes_num)
# Run the model
dim = 2
K = 4
bins_num = 3
prior_lambda = 1e5
batch_size = 100  #1
learning_rate = 0.001
epochs_num = 800  # 500
steps_per_epoch = 3
seed = utils.str2int("testing_prior")
verbose = True
shuffle = True


data = residual_events.get_pairs(), residual_events.get_events()
lm = LearningModel(
    data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,  last_time=1., batch_size=batch_size,
    prior_k=K, prior_lambda=prior_lambda,
    learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
    verbose=verbose, seed=seed
)
lm.learn()

# Predict the removed pairs

# Flatten the test events and pairs
flat_test_events = torch.as_tensor(
    [e for pair_events in residual_events.get_events() for e in pair_events], dtype=torch.float
)

flat_test_pairs = torch.repeat_interleave(
    torch.as_tensor(residual_events.get_pairs()),
    repeats=torch.as_tensor(list(map(len, residual_events.get_events())), dtype=torch.int),
    dim=0
).T

# Compute log-intensities
intensity_list = lm.get_log_intensity(
    times_list=flat_test_events,
    node_pairs=flat_test_pairs
).detach().numpy()


###########################
# Prediction experiment #
###########################
#   Let's remove the last 5% at random and predict them
split_time = 0.9
residual_events, removed_events = all_events.split_events_in_time(split_time=split_time)
removed_events.info()

# Construct samples
#   bins_num -> It divides the whole time interval into subintervals to sample events.
#   subsampling -> Instead of taking the whole event set as positive instances, a subsampling might be applied.
#                  The number of positive and negative samples are always equal.
#   with_time -> If it is True, it also samples time points for negative instances
true_labels, true_samples = removed_events.construct_samples(
    bins_num=2, subsampling=100, init_time=split_time, last_time=1.0, with_time=True
)
# Plot samples
all_events.plot_samples(labels=true_labels, samples=true_samples)


# Let's run the model
nodes_num = all_events.number_of_nodes()
print(f"Number of nodes: {nodes_num}")
# Run the model
dim = 2
K = 4
bins_num = 3
prior_lambda = 1e5
batch_size = 100  #1
learning_rate = 0.001
epochs_num = 800  # 500
steps_per_epoch = 3
seed = utils.str2int("testing_prior")
verbose = True
shuffle = True


data = residual_events.get_pairs(), residual_events.get_events()
lm = LearningModel(
    data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,  last_time=1., batch_size=batch_size,
    prior_k=K, prior_lambda=prior_lambda,
    learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
    verbose=verbose, seed=seed
)
lm.learn()

# Predict the removed pairs

# Flatten the test events and pairs
flat_test_events, flat_test_pairs = [], []
for triplet in true_samples:
    flat_test_events.append(triplet[2])
    flat_test_pairs.append([triplet[0], triplet[1]])

flat_test_events = torch.as_tensor(flat_test_events, dtype=torch.float)
flat_test_pairs = torch.as_tensor(flat_test_pairs, dtype=torch.int).T

# Compute log-intensities
intensity_list = lm.get_log_intensity(
    times_list=flat_test_events,
    node_pairs=flat_test_pairs
).detach().numpy()

fpr, tpr, _ = metrics.roc_curve(true_labels, intensity_list)
auc = metrics.roc_auc_score(true_labels, intensity_list)

# Plot ROC curve
plt.plot(fpr, tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()