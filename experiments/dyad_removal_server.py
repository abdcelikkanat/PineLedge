import os
import utils
import torch
from src.learning import LearningModel
from torch.utils.data import DataLoader
from src.events import Events
from sklearn.metrics import average_precision_score, roc_auc_score

# Set some parameters
dim = 2
K = 3
bins_num = 100
prior_weight_list = [1e10, 1e8, 1e6, 1e4, 1e2, 1e0] #1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0]  #[1e0, 1e2, 1e4, 1e6, 1e8, 1e10]
#batch_size = FULL
learning_rate = 0.01
epochs_num = 100  # 500
steps_per_epoch = 1
seed = utils.str2int("testing_dyad_sequential_dec_test")
verbose = True
shuffle = True
####
learn = True
###

dataset_name = f"3_clusters_mg_B=100_noise_s=0.1_rbf-s=-9.210290371559083_lambda=1.0_sizes=20_20_20_beta=1.5" #f"three_clusters_fp_sizes=15_20_10_beta=0"

# Define dataset and model path
#dataset_path = os.path.join(utils.BASE_FOLDER, "datasets", "synthetic", dataset_name)
dataset_path = os.path.join("/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/", dataset_name)

# Load the dataset
all_events = Events(seed=seed)
all_events.read(dataset_path)

batch_size = all_events.number_of_nodes()

# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)

# Get the number of nodes
nodes_num = all_events.number_of_nodes()

percent = 0.1
num = int(all_events.number_of_event_pairs() * percent)
residual, removed = all_events.remove_events(num=num, connected=False)

removed_pairs = removed.get_pairs()
removed_events = removed.get_events()

print(residual.number_of_nodes(), removed.number_of_nodes())

for pwIdx, prior_lambda in enumerate(prior_weight_list):
    print(f"PAIR: {pwIdx}")

    ###
    model_name = prev_model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pw={prior_weight_list[pwIdx]}" \
                              f"_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}_percent={percent}"

    #model_folder = os.path.join(utils.BASE_FOLDER, "experiments", "models", model_name)
    model_folder = os.path.join("/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/models", model_name)
    model_path = os.path.join(model_folder, f"{model_name}.model")

    # Construct data
    data = all_events.get_pairs(), all_events.get_events()
    # Run the model
    lm = LearningModel(data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,
                       prior_k=K, prior_lambda=prior_lambda,
                       #node_pairs_mask=torch.as_tensor(removed_pairs).T,
                       last_time=1.,
                       learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                       verbose=verbose, seed=seed)

    if learn:
        # Save the model
        os.makedirs(model_folder)

        # Load the previous model
        if pwIdx > 0:
            prev_model_name = f"{dataset_name}_D={dim}_B={bins_num}_K={K}_pw={prior_weight_list[pwIdx-1]}" \
                              f"_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}_percent={percent}"
            # lm.load_state_dict(torch.load(os.path.join(
            #     utils.BASE_FOLDER, "experiments", "models", prev_model_name, f"{prev_model_name}.model"
            # )))
            lm.load_state_dict(torch.load(os.path.join(
                "/Volumes/TOSHIBA EXT/RESEARCH/nikolaos/paper/pivem/models", prev_model_name, f"{prev_model_name}.model"
            )))

        lm.learn()
        torch.save(lm.state_dict(), model_path)
    else:
        lm.load_state_dict(torch.load(model_path))

    ####################################################################################################################
    threshold = 1
    test_intervals_num = 8
    test_intervals = torch.linspace(0, 1.0, test_intervals_num + 1)

    samples, labels = [[] for _ in range(test_intervals_num)], [[] for _ in range(test_intervals_num)]
    for i, j in utils.pair_iter(n=nodes_num):
        pair_events = torch.as_tensor(all_events[(i, j)][1])

        for b in range(test_intervals_num):
            test_bin_event_counts = torch.sum(
                (test_intervals[b + 1] > pair_events) * (pair_events >= test_intervals[b]))

            samples[b].append([i, j, test_intervals[b], test_intervals[b + 1]])

            if test_bin_event_counts >= threshold:
                labels[b].append(1)
            else:
                labels[b].append(0)

    # Predicted scores
    pred_scores = [[] for _ in range(test_intervals_num)]
    for b in range(test_intervals_num):
        for sample in samples[b]:
            i, j, t_init, t_last = sample
            score = lm.get_intensity_integral_for(i=i, j=j, interval=torch.as_tensor([t_init, t_last]))
            pred_scores[b].append(score)

    for b in range(test_intervals_num):
        roc_auc = roc_auc_score(y_true=labels[b], y_score=pred_scores[b])
        print(f"Roc AUC, Bin Id {b}: {roc_auc}")
        pr_auc = average_precision_score(y_true=labels[b], y_score=pred_scores[b])
        print(f"PR AUC, Bin Id {b}: {pr_auc}")
        print("")

    roc_auc_complete = roc_auc_score(
        y_true=[l for bin_labels in labels for l in bin_labels],
        y_score=[s for bin_scores in pred_scores for s in bin_scores]
    )
    print(f"Roc AUC in total: {roc_auc_complete}")
    pr_auc_complete = average_precision_score(
        y_true=[l for bin_labels in labels for l in bin_labels],
        y_score=[s for bin_scores in pred_scores for s in bin_scores]
    )
    print(f"PR AUC in total: {pr_auc_complete}")

    intenstiy_integral_sum = 0
    print(f"Integral sum: {sum([s for bin_scores in pred_scores for s in bin_scores])}")
