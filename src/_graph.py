import torch
import random
import numpy as np
import pickle as pkl
from torch.utils.data.dataset import Dataset


class CustomizedDataset(Dataset):

    def __init__(self, file_path, init_time=0, last_time=None, time_normalization=True, shuffle=False,
                 train_ratio=1.0, seed=123):

        super().__init__()

        self.__init_time = init_time
        self.__last_time = last_time

        self.__first_event = None
        self.__last_event = None

        self.__file_path = file_path
        self.__shuffle = shuffle
        self.__train_ratio = train_ratio

        # Load the dataset
        self.__events, self.__num_of_nodes, self.__node_pairs, node2group = self.__load()

        if time_normalization:
            self.__normalize()

        self.__events_train, self.__events_test = self.__split_train_test()

        self.__idx2pair = {
            idx: pair for idx, pair in enumerate(
                [(i, j) for i in range(self.__num_of_nodes) for j in range(i+1, self.__num_of_nodes)]
            )
        }

    def __load(self):

        with open(self.__file_path, 'rb') as f:

            data = pkl.load(f)
            events = data["events"]
            num_of_nodes = data.get("n", len(data["events"][0].keys()) + 1)
            node_pairs = [
                [i for i in range(num_of_nodes) for _ in range(i + 1, num_of_nodes)],
                [j for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)],
            ]

        self.__first_event = +1e6
        self.__last_event = -1e6
        for i, j in zip(node_pairs[0], node_pairs[1]):

            for t in events[i][j]:
                if t > self.__last_event:
                    self.__last_event = t
                if t < self.__first_event:
                    self.__first_event = t

        # Set the initial and the last point of the timeline
        if self.__init_time is None:
            self.__init_time = self.__first_event
        if self.__last_time is None:
            self.__last_time = self.__last_event

        # Get the node groups
        node2group = data.get("node2group", None)

        return events, num_of_nodes, node_pairs, node2group

    def __normalize(self):

        timeline_len = float(self.__last_time - self.__init_time)
        for i, j in zip(self.__node_pairs[0], self.__node_pairs[1]):
            for t_idx, t in enumerate(self.__events[i][j]):
                self.__events[i][j][t_idx] = (t - self.__init_time) / timeline_len

            # print(i, j, len(self.__events[i][j]), )
            # if len(self.__events[i][j]):
            #     print(min(self.__events[i][j]), max(self.__events[i][j]))

        self.__last_time = 1.0
        self.__init_time = 0.0

    def __split_train_test(self):

        # Construct the event lists for training and testing sets
        events_train = {i: {j: [] for j in range(i+1, self.__num_of_nodes)} for i in range(self.__num_of_nodes)}
        events_test = {i: {j: [] for j in range(i+1, self.__num_of_nodes)} for i in range(self.__num_of_nodes)}
        for i, j in zip(self.__node_pairs[0], self.__node_pairs[1]):

            for t in self.__events[i][j]:
                if t < self.get_last_time(set_type="train"):
                    events_train[i][j].append(t)
                else:
                    events_test[i][j].append(t)

        return events_train, events_test

    def __getitem__(self, idx):

        u, v = self.__idx2pair[idx]
        return u, v, self.__events_train[u][v]

    def __len__(self):

        return self.__num_of_nodes * (self.__num_of_nodes - 1) // 2

    def get_num_of_nodes(self):

        return self.__num_of_nodes

    def get_init_time(self, set_type="all"):

        if set_type == "train" or set_type == "all":

            return self.__init_time

        elif set_type == "test":

            return self.__last_time * self.__train_ratio

        else:

            raise ValueError("Invalid set type!")

    def get_last_time(self, set_type="all"):

        if set_type == "train":

            return self.__last_time * self.__train_ratio

        elif set_type == "test" or set_type == "all":

            return self.__last_time

        else:

            raise ValueError("Invalid set type!")

    def get_first_event_time(self, set_type="all"):

        if set_type == "train" or set_type == "test":
            raise ValueError("Not implemented!")

        return self.__first_event

    def get_last_event_time(self, set_type="all"):

        if set_type == "train" or set_type == "test":
            raise ValueError("Not implemented!")

        return self.__last_event

    def get_groups(self):

        if self.__node2group is None:
            return None

        return list(map(self.__node2group.get, range(self.__num_of_nodes)))

    def get_train_data(self):

        return self.__events_train

    def get_test_data(self):

        return self.__events_test


class DynamicGraph:

    def __init__(self, seed=0):

        self.__nodes_num = None
        self.__edges = None
        self.__events = None

        # Set the seed value
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def read_pkl(self, file_path):

        with open(file_path, 'rb') as f:

            data = pkl.load(f)
            events = data["events"]

        # Get the number of nodes
        self.__nodes_num = data.get("n", len(data["events"][0].keys()) + 1)
        # The nodes are integer values
        self.__nodes = [node for node in range(self.__nodes_num)]
        # Events are a dictionary of a dictionary storing a list of events
        self.__events = events
        # Edges
        self.__edges = []
        for i in events.keys():
            for j in events[i].keys():
                if len(events[i][j]) == 0:
                    self.__edges.append((i, j))

    def write_pkl(self, file_path):

        data = {'n': self.__nodes_num, 'events': self.__events}

        with open(file_path, 'wb') as f:
            pkl.dump(data, f)

    def pairs(self):

        for i in range(self.__nodes_num):
            for j in range(i+1, self.__nodes_num):
                yield i, j

    def __getitem__(self, item1, item2=None):

        if item2 is None:
            return self.__events[item1]
        else:
            return self.__events[item1][item2]

    def number_of_nodes(self):

        return self.__nodes_num

    def number_of_edges(self):

        return len(self.__edges)

    def number_of_events(self):

        return sum(len(self.__events[pair[0]][pair[1]]) for pair in self.pairs())

    def nodes(self):

        return self.__nodes

    def edges(self):

        return self.__edges

    def info(self):

        print(f"Number of nodes: {self.number_of_nodes()}")
        print(f"Number of edges: {self.number_of_edges()}")
        print(f"Number of events: {self.number_of_events()}")

    # def __load(self):
    #
    #
    #
    #     self.__first_event = +1e6
    #     self.__last_event = -1e6
    #     for i, j in zip(node_pairs[0], node_pairs[1]):
    #
    #         for t in events[i][j]:
    #             if t > self.__last_event:
    #                 self.__last_event = t
    #             if t < self.__first_event:
    #                 self.__first_event = t
    #
    #     # Set the initial and the last point of the timeline
    #     if self.__init_time is None:
    #         self.__init_time = self.__first_event
    #     if self.__last_time is None:
    #         self.__last_time = self.__last_event
    #
    #     # Get the node groups
    #     node2group = data.get("node2group", None)
    #
    #     return events, num_of_nodes, node_pairs, node2group
    #
    # def get_number_of_nodes(self):
    #
    #     return self.__nodes_num


g = DynamicGraph()
g.read_pkl(file_path="../datasets/real/ia_enron/ia_enron_events.pkl")
g.info()
