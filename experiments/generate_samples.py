import os
import sys
import pickle
sys.path.insert(0, '.')
from src.events import Events
from src.experiments import Experiments

## Parameters
seed = 123
time_normalization = True
init_time = 0.0
last_time = 1.0
split_time = 1.0 #0.8
####

#dataset = "ia_enron"
#dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', "datasets", "real", dataset, f"{dataset}_events.pkl")
dataset = "three_clusters"
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', "datasets", "synthetic", f"{dataset}_events.pkl")

# Read the events
events = Events(seed=seed)
events.read(dataset_path)
# Normalize the timeline to [0, 1]
if time_normalization:
    events.normalize(init_time=init_time, last_time=last_time)
else:
    init_time = events.get_first_event_time()
    last_time = events.get_last_event_time()
# train_events = events.get_train_events(last_time=split_time)
# test_events = events.get_test_events(init_time=split_time)


bins_num = 20
subsampling = 0
with_time = 1
exp = Experiments(seed=seed)
# exp.set_events(events=train_events)
# train_labels, train_samples = exp.construct_samples(
#     bins_num=int(bins_num*split_time), subsampling=subsampling, with_time=with_time, init_time=init_time, last_time=split_time
# )
# exp.set_events(events=test_events)
# test_labels, test_samples = exp.construct_samples(
#     bins_num=int(bins_num*(last_time-split_time)), subsampling=subsampling, with_time=with_time, init_time=split_time, last_time=last_time
# )
exp.set_events(events=events)
train_labels, train_samples = exp.construct_samples(
    bins_num=int(bins_num*split_time), subsampling=subsampling, with_time=with_time, init_time=init_time, last_time=split_time
)
test_labels, test_samples = exp.construct_samples(
    bins_num=int(bins_num*(1.0-split_time)), subsampling=subsampling, with_time=with_time, init_time=split_time, last_time=last_time
)

exp.plot_samples(labels=train_labels, samples=train_samples)
# exp.plot_samples(labels=test_labels, samples=test_samples)


filename = f"{dataset}_tInit={init_time}_tLast={last_time}_sTime={split_time}_binNum={bins_num}_subsampling={subsampling}"
sample_list_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'samples', f"{filename}_list.pkl")
sample_dict_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'samples', f"{filename}_dict.pkl")

with open(sample_list_file_path, "wb") as f:
    pickle.dump({"train": {"labels": train_labels, "samples": train_samples},
                 "test": {"labels": test_labels, "samples": test_samples}}, f)

nodes_num = events.number_of_nodes()
# train_samples_dict, train_labels_dict, test_samples_dict, test_labels_dict = dict(), dict(), dict(), dict()
# for i in range(nodes_num):
#     train_samples_dict[i] = dict()
#     train_labels_dict[i] = dict()
#     for j in range(i + 1, nodes_num):
#         train_samples_dict[i][j] = []
#         train_labels_dict[i][j] = []
#         test_samples_dict[i][j] = []
#         test_labels_dict[i][j] = []

train_samples_dict = {i: {j: [] for j in range(i+1, nodes_num)} for i in range(nodes_num)}
train_labels_dict = {i: {j: [] for j in range(i+1, nodes_num)} for i in range(nodes_num)}
test_samples_dict = {i: {j: [] for j in range(i+1, nodes_num)} for i in range(nodes_num)}
test_labels_dict = {i: {j: [] for j in range(i+1, nodes_num)} for i in range(nodes_num)}


for sample, label in zip(train_samples, train_labels):
    train_samples_dict[sample[0]][sample[1]].append(sample[2])
    train_labels_dict[sample[0]][sample[1]].append(label)
for sample, label in zip(test_samples, test_labels):
    test_samples_dict[sample[0]][sample[1]].append(sample[2])
    test_labels_dict[sample[0]][sample[1]].append(label)

with open(sample_dict_file_path, "wb") as f:
    pickle.dump({"train": {"labels": train_labels_dict, "samples": train_samples_dict},
                "test": {"labels": test_labels_dict, "samples": test_samples_dict}}, f)

