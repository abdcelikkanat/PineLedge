import os
import sys
import torch
sys.path.insert(0, '.')
sys.path.insert(1, '..')
import utils
from src.learning import LearningModel
from torch.utils.data import DataLoader
from src.events import Events

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# Define the parameters #
dim = 2
bins_num = 10
learning_rate = 0.1
epochs_num = 150 #100  # 500
seed = 666661

batch_size = 9 #resistance_game=4 = 9  #151  # Lyon=242 # # ia_enron=151 # ia_contacts=113
steps_per_epoch = 5
shuffle = True
verbose = True

time_normalization = 1
init_time = 0.0
last_time = 1.0
split_time = 0.8

datasetname = "resistance_game=4" #"4nodedense" #"resistance_game=4" #"ia_enron"  # resistance_game=4
#########################

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', "datasets", "real", datasetname, f"{datasetname}_events.pkl")
# dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', "datasets", "synthetic", f"{datasetname}_events.pkl")

modelname = f"{datasetname}_scaled={time_normalization}_split={split_time}_D={dim}_B={bins_num}_lr={learning_rate}_epoch={epochs_num}_batch_size={batch_size}_batch_steps={steps_per_epoch}_seed={seed}"


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
test_events = events.get_test_events(init_time=split_time, batch_size=batch_size)

# Get the number of nodes in the whole dataset
nodes_num = events.number_of_nodes()
print(f"Number of nodes in the events set: {nodes_num}")
print(f"Number of events in the training set: {train_events.number_of_events()}")
print(f"Number of events in the testing set: {test_events.number_of_events()}")

# Load the dataset
data_loader = DataLoader(train_events, batch_size=1, shuffle=shuffle, collate_fn=utils.collate_fn)
prior_weights = [0., 0.0001, 0.001, 0.01, 0.1, 1.0] #[-1.0]
for pw in prior_weights:
    print(f"Weight: {pw}")
    # Run the model
    lm = LearningModel(data_loader=data_loader, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=split_time,
                       learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch, device=device,
                       verbose=verbose, seed=seed, prior_weight=pw)
    lm.to(device)
    # Run the model
    lm.learn()
    # Model path
    model_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   f"../experiments/models/prior_analysis_{modelname}_pw={pw}.model")
    # Save the model
    torch.save(lm.state_dict(), model_file_path)






