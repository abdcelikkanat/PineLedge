import os
import torch
import pickle
from argparse import ArgumentParser, RawTextHelpFormatter
from sklearn.metrics import average_precision_score, roc_auc_score
import utils
from src.events import Events
from src.learning import LearningModel

########################################################################################################################
parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--samples_folder', type=str, required=True, help='Path of the samples folder'
)
parser.add_argument(
    '--samples_set', type=str, required=True, choices=["test", "valid"], help='Path of the samples folder'
)
parser.add_argument(
    '--model_path', type=str, required=True, help='Path of the model'
)
parser.add_argument(
    '--output_path', type=str, required=True, help='Path of the output file'
)

args = parser.parse_args()
########################################################################################################################

seed = 19
# Set some parameters
samples_folder = args.samples_folder
samples_set = args.samples_set
model_path = args.model_path
output_path = args.output_path

########################################################################################################################

print("+ Model is being read...")
# Load the model
kwargs, lm_state = torch.load(model_path, map_location=torch.device('cpu'))
kwargs['device'] = 'cpu'
lm = LearningModel(**kwargs)
lm.load_state_dict(lm_state)
print("\t+ Completed.")

########################################################################################################################

# Read samples file
print("+ Samples are being read...")
pos_samples_file_path = os.path.join(samples_folder, f"{samples_set}_pos.samples")
with open(pos_samples_file_path, 'rb', ) as f:
    pos_samples = pickle.load(f)

neg_samples_file_path = os.path.join(samples_folder, f"{samples_set}_neg.samples")
with open(neg_samples_file_path, 'rb', ) as f:
    neg_samples = pickle.load(f)
print("\t+ Completed.")

########################################################################################################################

y_true = []
y_pred = []

print("+ The experiment has been initialized.")
for samples in pos_samples:
    i, j, t_init, t_last = samples
    y_true.append(1)
    y_pred.append(
        lm.get_intensity_integral_for(i=i, j=j, interval=torch.as_tensor([t_init, t_last])).detach().numpy()
    )

for samples in neg_samples:
    i, j, t_init, t_last = samples

    y_true.append(0)
    y_pred.append(
        lm.get_intensity_integral_for(i=i, j=j, interval=torch.as_tensor([t_init, t_last])).detach().numpy()
    )

with open(output_path, 'w') as f:
    roc_auc_complete = roc_auc_score(y_true=y_true, y_score=y_pred)
    f.write(f"Roc_AUC: {roc_auc_complete}\n")
    pr_auc_complete = average_precision_score(y_true=y_true, y_score=y_pred)
    f.write(f"PR_AUC: {pr_auc_complete}\n")
    events_count, alpha1, alpha2 = lm.compute_coefficients(
        kwargs["nodes_num"], kwargs["data"][0], kwargs["data"][1], kwargs["bins_num"]
    )
    nll = lm.get_negative_log_likelihood(events_count, alpha1, alpha2)
    f.write(f"NLL: {nll}\n")


print("\t+ Roc_AUC: {}".format(roc_auc_complete))
print("\t+ PR_AUC: {}".format(pr_auc_complete))
print("\t+ Completed.")

