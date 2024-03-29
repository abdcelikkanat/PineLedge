import torch
from src.learning import LearningModel
from visualization.animation import Animation
from src.events import Events
import os
import sys
import utils
import pickle as pkl

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


# Set some paremeters
dim = 2
K = 2
bins_num = 3
prior_lambda = 1e5
batch_size = 2  #1
learning_rate = 0.1
epochs_num = 600  # 500
steps_per_epoch = 5
seed = 123
verbose = True
time_normalization = True
shuffle = True
visualization = True
suffix = ""

dataset_name = "four_nodes_fp"
model_name = f"{dataset_name}_D={dim}_B={bins_num}_k={K}_pl={prior_lambda}_lr={learning_rate}_e={epochs_num}_spe={steps_per_epoch}_s={seed}"+suffix

# Define dataset and model path
dataset_folder = os.path.join(
    utils.BASE_FOLDER, "datasets", "synthetic", dataset_name
)
dataset_path = os.path.join(
    dataset_folder, f"{dataset_name}_events.pkl"
)
model_folder = os.path.join(
    utils.BASE_FOLDER, "experiments", "models", model_name
)
model_path = os.path.join(
    model_folder, f"{model_name}.model"
)
anim_path = os.path.join(
    utils.BASE_FOLDER, "experiments", "animations", f"{model_name}.mp4"
)

# Load the dataset
all_events = Events(seed=seed)
all_events.read(dataset_path)

# Normalize the events
all_events.normalize(init_time=0, last_time=1.0)

# Get the number of nodes
nodes_num = all_events.number_of_nodes()

data = all_events.get_pairs(), all_events.get_events()

# Run the model
lm = LearningModel(data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim,  batch_size=batch_size, last_time=1.,
                   prior_k=K, prior_lambda=prior_lambda,
                   learning_rate=learning_rate, epochs_num=epochs_num, steps_per_epoch=steps_per_epoch,
                   verbose=verbose, seed=seed)
#
# s = lm.get_xt(
#     events_times_list=torch.as_tensor([0.73,]),
#     x0=torch.as_tensor([[0.3, 0.6]]),
#     v=torch.as_tensor([[[0.2, 0.2]], [[0.2, 0.2]], [[0.2, 0.2]]]),
# )
# print("> ", s)

lm.learn()

if visualization:

    frame_times = torch.linspace(0, 1.0, 100)
    # Ground truth animation
    # bm = BaseModel(x0=x0, v=v, beta=beta, last_time=last_time, bins_rwidth=bins_rwidth)
    # embs_gt = bm.get_xt(times_list=times_list*last_time).detach().numpy()
    # node2color = [0] * cluster_sizes[0] + [1] * cluster_sizes[1] + [2] * cluster_sizes[2]
    # anim = Animation(embs_gt, fps=12, node2color=node2color)
    # anim.save("./three_clusters_gt_sil.mp4")

    # Read the group information
    node2group_filepath = os.path.join(dataset_folder, f"{dataset_name}_node2group.pkl")
    with open(node2group_filepath, "rb") as f:
        node2group_data = pkl.load(f)
    node2group, group2node = node2group_data["node2group"], node2group_data["group2node"]

    embs_pred = lm.get_xt(
        events_times_list=torch.cat([frame_times]*lm.get_number_of_nodes()),
        x0=torch.repeat_interleave(lm.get_x0(), repeats=len(frame_times), dim=0),
        v=torch.repeat_interleave(lm.get_v(), repeats=len(frame_times), dim=1)
    ).reshape((lm.get_number_of_nodes(), len(frame_times),  lm.get_dim())).transpose(0, 1).detach().numpy()

    node2color = [node2group[idx] for idx in range(nodes_num)]
    anim = Animation(embs_pred, fps=12, node2color=node2color,
                     data=data, frame_times=frame_times.numpy())
    anim.save(anim_path)
