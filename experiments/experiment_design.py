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
from scipy.stats import spearmanr
import numpy as np

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
p = 0.05

# Define dataset
sample_bins = 16
dataset_folder = os.path.join(
    utils.BASE_FOLDER, "experiments", f"samples_reconst_bins={sample_bins}", f"{dataset_name}_p={p}_normalized"
)

# Load the dataset
residual_events = Events(path=dataset_folder, seed=seed)
# Print dataset info
residual_events.info()
# # Normalize the events # We don't need it
# all_events.normalize(init_time=0, last_time=1.0)

# Load sample files
with open(os.path.join(dataset_folder, "samples.pkl"), 'rb') as f:
    samples_data = pkl.load(f)
    #valid_labels = samples_data["valid_labels"]
    #valid_samples = samples_data["valid_samples"]
    test_labels = samples_data["test_labels"]
    test_samples = samples_data["test_samples"]

# Write the results
with open("./results.txt", 'w') as fout:

    for epochs_num in [20, ]: #[100, 300]: #[60, 120, 360, 720, 1024]:
        for bins_num in [10, ]: #[128, 64, 16, 4, 1]: #[1, 4, 16, 64, 128]:

            fout.write(f"Epoch: {epochs_num}, Bin: {bins_num}\n")

            # Let's run the model
            nodes_num = residual_events.number_of_nodes()
            # Run the model
            dim = 2
            K = 4
            #bins_num = 10
            prior_lambda = 1e5
            batch_size = nodes_num  #1
            learning_rate = 0.001
            #epochs_num = 360  # 500
            steps_per_epoch = 1
            seed = utils.str2int("testing_reconstruction")
            verbose = True
            learn = True

            # Construct the data for the learning model
            residual_data = residual_events.get_pairs(), residual_events.get_events()
            # Initialize the constructor
            lm = LearningModel(
                data=residual_data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,  last_time=1., batch_size=batch_size,
                prior_k=K, prior_lambda=prior_lambda,
                learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                verbose=verbose, seed=seed,
            )
            # Learn the model
            model_name = f"{dataset_name}_sample-bins={sample_bins}_D={dim}_B={bins_num}_K={K}_pl={prior_lambda}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}"
            model_path = os.path.join(utils.BASE_FOLDER, "experiments", "models", f"{model_name}.model")
            if learn:
                lm.learn()
                torch.save(lm.state_dict(), model_path)

            else:
                lm.load_state_dict(torch.load(model_path))

            bin_pred_scores = [[] for _ in range(sample_bins)]
            bin_test_labels = [[] for _ in range(sample_bins)]
            bin_test_samples = [[] for _ in range(sample_bins)]
            for sample, label in zip(test_samples, test_labels):

                pred = lm.get_intensity_integral_for(
                    i=sample[0], j=sample[1],
                    interval=torch.as_tensor([sample[2], sample[3]])
                )

                bin_idx = int(sample[2] / (1.0 / sample_bins))

                bin_pred_scores[bin_idx].append(pred)
                bin_test_samples[bin_idx].append(sample)
                bin_test_labels[bin_idx].append(label)

            nmis_avg, amis_avg, auc_avg, corr_avg = 0., 0., 0., 0.
            zero_bin_counter = 0
            for bin_idx in range(sample_bins):

                if sum(bin_test_labels[bin_idx]) == 0 or sum(bin_test_labels[bin_idx]) == len(bin_test_labels[bin_idx]):
                    zero_bin_counter += 1

                else:

                    fout.write(f"Sample Bin Idx: {bin_idx}\n")


                    # Compute some metrics
                    nmis = normalized_mutual_info_score(labels_true=bin_test_labels[bin_idx], labels_pred=bin_pred_scores[bin_idx])
                    amis = adjusted_mutual_info_score(labels_true=bin_test_labels[bin_idx], labels_pred=bin_pred_scores[bin_idx])
                    auc = roc_auc_score(y_true=bin_test_labels[bin_idx], y_score=bin_pred_scores[bin_idx])

                    corr = spearmanr(np.asarray(bin_test_samples[bin_idx], dtype=np.float), np.asarray(bin_pred_scores[bin_idx], dtype=np.float))
                    fout.write(f"Normalized Mutual Info Score: {nmis}\n")
                    fout.write(f"Adjusted Mutual Info Score: {amis}\n")
                    fout.write(f"ROC AUC Score: {auc}\n")
                    fout.write(f"Spearmanr: {corr}\n")

                    nmis_avg += nmis
                    amis_avg += amis
                    auc_avg += auc
                    # corr_avg += corr

        fout.write("Avg\n")
        fout.write(f"Avg Normalized Mutual Info Score: {nmis_avg/float(sample_bins-zero_bin_counter)}\n")
        fout.write(f"Avg Adjusted Mutual Info Score: {amis_avg/float(sample_bins-zero_bin_counter)}\n")
        fout.write(f"Avg ROC AUC Score: {auc_avg/float(sample_bins-zero_bin_counter)}\n")
        # fout.write(f"Avg Spearmanr: {corr_avg/float(sample_bins-zero_bin_counter)}\n")