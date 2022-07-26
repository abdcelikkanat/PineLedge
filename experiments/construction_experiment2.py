import torch
from argparse import ArgumentParser, RawTextHelpFormatter
from sklearn.metrics import average_precision_score, roc_auc_score
import utils
import os
import pickle
from src.events import Events
from src.learning import LearningModel

########################################################################################################################
parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)

parser.add_argument(
    '--samples_folder', type=str, required=True, help='Path of the samples'
)
parser.add_argument(
    '--model_path', type=str, required=True, help='Path of the model'
)
parser.add_argument(
    '--output_path', type=str, required=True, help='Path of the output file'
)
# Set some parameters
args = parser.parse_args()
seed = 19
samples_folder = args.samples_folder
model_path = args.model_path
output_path = args.output_path

########################################################################################################################

print("+ Model is being read...")
# Load the model
kwargs, lm_state = torch.load(model_path, map_location=torch.device('cpu'))
kwargs['device'] = 'cpu'
lm = LearningModel(**kwargs)
lm.load_state_dict(lm_state)
print("\t- Completed.")

########################################################################################################################

pos_samples_file = os.path.join(samples_folder, 'pos.samples')
with open(pos_samples_file, 'rb') as f:
    pos_samples = pickle.load(f)
neg_samples_file = os.path.join(samples_folder, 'neg.samples')
with open(neg_samples_file, 'rb') as f:
    neg_samples = pickle.load(f)

########################################################################################################################
# Load the dataset
all_events = Events(seed=seed)
all_events.read(os.path.join(utils.BASE_FOLDER, 'datasets', 'real', 'ia-contact'))
# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)
# Get the number of nodes
nodes_num = all_events.number_of_nodes()
print("Nodes num: ", nodes_num)
# file = open(output_path, 'w')
# print("Printing...")
#
# # Load the dataset
# all_events = Events(seed=seed)
# all_events.read(dataset_folder)
# # Normalize the events
# all_events.normalize(init_time=0, last_time=1.0)
# # Get the number of nodes
# nodes_num = all_events.number_of_nodes()
#
# file = open(output_path, 'w')
# threshold = 1
# test_intervals = torch.linspace(0, 1.0, test_intervals_num+1)
#
# all_labels = []
# all_pred_scores = []
# for b in range(test_intervals_num):
#
#     labels = []
#     pred_scores = []
#
#     print(f"+ Bin id: {b+1}/{test_intervals_num}")
#
#     for idx, pair in enumerate(utils.pair_iter(n=nodes_num, undirected=True)):
#
#         if idx % int(nodes_num * (nodes_num-1) / 2 / 10) == 0:
#             print("\t- {:.2f}% completed.".format(idx * 100 / (nodes_num * (nodes_num-1) / 2)))
#
#         i, j = pair
#         pair_events = torch.as_tensor(all_events[(i, j)][1])
#
#         test_bin_event_counts = torch.sum( (test_intervals[b+1] > pair_events) * (pair_events >= test_intervals[b]) )
#
#         labels.append(
#             1 if test_bin_event_counts >= threshold else 0
#         )
#
#         pred_scores.append(
#             lm.get_intensity_integral_for(i=i, j=j, interval=torch.as_tensor([test_intervals[b], test_intervals[b+1]])).detach().numpy()
#         )
#
#     if sum(labels) != len(labels) and sum(labels) != 0:
#
#         roc_auc = roc_auc_score(y_true=labels, y_score=pred_scores)
#         file.write(f"Roc_AUC_Bin_Id_{b}: {roc_auc}\n")
#         pr_auc = average_precision_score(y_true=labels, y_score=pred_scores)
#         file.write(f"PR_AUC_Bin_Id_{b}: {pr_auc}\n")
#         file.write("\n")
#
#     all_labels.extend(labels)
#     all_pred_scores.extend(pred_scores)

y_score = []
y_true = []
for sample in pos_samples:
    i, j, t_init, t_last = sample
    value = lm.get_intensity_integral_for(
        i=i, j=j, interval=torch.as_tensor([t_init, t_last])
    ).detach().numpy()
    y_score.append(value)
    y_true.append(1)

for sample in neg_samples:
    i, j, t_init, t_last = sample
    value = lm.get_intensity_integral_for(
        i=i, j=j, interval=torch.as_tensor([t_init, t_last])
    ).detach().numpy()
    y_score.append(value)
    y_true.append(0)



roc_auc = roc_auc_score(
    y_true=y_true,
    y_score=y_score
)
# file.write(f"Roc_AUC_total: {roc_auc_complete}\n")
pr_auc = average_precision_score(
    y_true=y_true,
    y_score=y_score
)
print(roc_auc, pr_auc)
# file.write(f"PR_AUC_total: {pr_auc_complete}\n")
#
# file.close()

