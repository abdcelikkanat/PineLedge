import os
import utils
import torch
from src.learning import LearningModel
from src.events import Events
import networkx as nx
import pickle as pkl

# Dataset name
dataset_name = f"soc-wiki-elec" #f"soc-sign-bitcoinalpha" #f"soc-wiki-elec"
# Define dataset
dataset_folder = os.path.join(utils.BASE_FOLDER, "datasets", "real", dataset_name)

# Load the dataset
all_events = Events(path=dataset_folder, seed=0)

g = nx.Graph()
g.add_edges_from(all_events.get_pairs())

if nx.is_connected(g):
    print("Connected!")

else:

    print("Not connected!")

    gcc_nodes = max(nx.connected_components(g), key=len)
    gcc = g.subgraph(gcc_nodes)

    node2newlabel = {node: newlabel for newlabel, node in enumerate(gcc.nodes())}
    new_pairs = []
    new_events = []
    for pair, events in zip(all_events.get_pairs(), all_events.get_events()):
        if gcc.has_edge(pair[0], pair[1]):
            u, v = node2newlabel[pair[0]], node2newlabel[pair[1]]
            new_pairs.append([u, v] if u < v else [v, u])
            new_events.append(events)

    new_dataset_folder = os.path.join(utils.BASE_FOLDER, "datasets", "real", dataset_name+"_new")
    if not os.path.exists(new_dataset_folder):
        os.makedirs(new_dataset_folder)

    with open(os.path.join(new_dataset_folder, "events.pkl"), 'wb') as f:
        pkl.dump(new_events, f)

    with open(os.path.join(new_dataset_folder, "pairs.pkl"), 'wb') as f:
        pkl.dump(new_pairs, f)

    print(all_events.number_of_nodes(), gcc.number_of_nodes())