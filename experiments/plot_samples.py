import os
import sys
import pickle
sys.path.insert(0, '.')
from src.events import Events
from src.experiments import Experiments
from sklearn.metrics import roc_auc_score

seed = 123
time_normalization = True
init_time = 0.0
last_time = 1.0
split_time = 0.8

dataset = "ia_enron" #"resistance_game=4"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', "datasets", "real", dataset, f"{dataset}_events.pkl")

# Read the events
events = Events(seed=seed)
events.read(dataset_path)
# Normalize the timeline to [0, 1]
if time_normalization:
    events.normalize(init_time=init_time, last_time=last_time)
else:
    init_time = events.get_first_event_time()
    last_time = events.get_last_event_time()
train_events = events.get_train_events(last_time=split_time)
# test_events = events.get_test_events(init_time=split_time)


bins_num = 20
subsampling = 0
with_time = 1
exp = Experiments(seed=seed)
exp.set_events(events=events)

filename = f"{dataset}_tInit={init_time}_tLast={last_time}_sTime={split_time}_binNum={bins_num}_subsampling={subsampling}"
sample_list_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'samples', f"{filename}_list.pkl")
sample_dict_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'samples', f"{filename}_dict.pkl")

with open(sample_list_file_path, "rb") as f:
    sample_list = pickle.load(f)

train_labels, train_samples = sample_list["train"]["labels"], sample_list["train"]["samples"]
test_labels, test_samples = sample_list["test"]["labels"], sample_list["test"]["samples"]

nodes_num = events.number_of_nodes()


exp.plot_samples(labels=train_labels, samples=train_samples, figsize=(4, 12))
# exp.plot_samples(labels=test_labels, samples=test_samples)

exp.set_events(events=train_events)
F = exp.get_freq_map()
simon_auc = roc_auc_score(y_true=train_labels, y_score=[F[sample[0], sample[1]] for sample in train_samples])
print(f"Simon training auc score: {simon_auc}")

simon_auc = roc_auc_score(y_true=test_labels, y_score=[F[sample[0], sample[1]] for sample in test_samples])
print(f"Simon testing auc score: {simon_auc}")