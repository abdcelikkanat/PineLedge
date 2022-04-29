import os
import math
import random
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from math import factorial as fac
from utils.utils import linearIdx2matIdx
import copy


class Events(Dataset):

    def __init__(self, data=(None, None), nodes_num=None, seed=0):

        super().__init__()

        self.__nodes_num = nodes_num
        self.__events_dict = None

        self.__events = data[0]
        self.__pairs = data[1]

        self.__nodes = None
        # Set the seed value
        self.__set_seed(seed=seed)

    def __set_seed(self, seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def read(self, file_path):

        with open(file_path, 'rb') as f:
            data = pkl.load(f)

        self.__initialize(
            events_dict=data["events"], nodes_num=data.get("n", len(data["events"][0].keys()) + 1)
        )

        self.__events, self.__pairs = [], []
        for i in range(self.__nodes_num):
            for j in range(i+1, self.__nodes_num):

                if len(self.__events_dict[i][j]):
                    self.__pairs.append([i, j])
                    self.__events.append(self.__events_dict[i][j])

    def __initialize(self, events_dict, nodes_num):
        # Events are a dictionary of a dictionary storing a list of events
        self.__events_dict = events_dict
        # Get the number of nodes
        self.__nodes_num = nodes_num
        # The nodes are integer values
        self.__nodes = [node for node in range(self.__nodes_num)]

    def write(self, file_path):

        data = {'events': self.__events, 'n': self.__nodes_num}

        with open(file_path, 'wb') as f:
            pkl.dump(data, f)

    # def pairs(self):
    #
    #     for i in range(self.__nodes_num):
    #         for j in range(i+1, self.__nodes_num):
    #             yield i, j

    def number_of_nodes(self):

        return self.__nodes_num

    def number_of_event_pairs(self):

        return len(self.__events)

    def number_of_total_events(self):

        return sum(len(events) for events in self.__events)

    def nodes(self):

        return self.__nodes

    def get_data_dict(self):

        return self.__events_dict

    def get_events(self):

        return self.__events

    def get_pairs(self):

        return self.__pairs

    def normalize(self, init_time=0, last_time=None):

        min_event_time = min([min(pair_events) for pair_events in self.__events])
        max_event_time = max([max(pair_events) for pair_events in self.__events])

        # for i, j in self.pairs():
        #
        #     events = self.__events[i][j]
        #     if len(events) > 0:
        #         min_value, max_value = min(events), max(events)
        #
        #         if max_value > last_event_time:
        #             last_event_time = max_value
        #         if min_value < first_event_time:
        #             first_event_time = min_value
        #
        # # Set the last time if not set
        # last_time = 1.0 if last_time is None else last_time
        #
        # a = (last_time - init_time) / (last_event_time - first_event_time)
        # b = init_time - a * first_event_time
        # for i, j in self.pairs():
        #     for idx in range(len(self.__events[i][j])):
        #         self.__events[i][j][idx] = a * self.__events[i][j][idx] + b

        for i in range(len(self.__events)):
            for j in range(len(self.__events[i])):
                self.__events[i][j] = (self.__events[i][j] - min_event_time) / (max_event_time - min_event_time)

    def remove_events(self, num):

        chosen_indices = np.random.choice(len(self.__events), size=(num,), replace=False)

        residual_events = []
        residual_pairs = []
        removed_events = []
        removed_pairs = []

        for idx in range(len(self.__events)):
            if idx in chosen_indices:
                removed_events.append(self.__events[idx])
                removed_pairs.append(self.__pairs[idx])
            else:
                residual_events.append(self.__events[idx])
                residual_pairs.append(self.__pairs[idx])

        return Events(data=(residual_events, residual_pairs)), Events(data=(removed_events, removed_pairs))

    # def get_first_event_time(self):
    #
    #     min_value = +1e6
    #     for i, j in self.pairs():
    #         events = self.__events[i][j]
    #         if len(events) > 0:
    #             temp = min(events)
    #             if min_value > temp:
    #                 min_value = temp
    #
    #     return min_value
    #
    # def get_last_event_time(self):
    #
    #     max_value = -1e6
    #     for i, j in self.pairs():
    #         events = self.__events[i][j]
    #         if len(events) > 0:
    #             temp = max(events)
    #             if max_value < temp:
    #                 max_value = temp
    #
    #     return max_value

    # def get_train_events(self, last_time, batch_size=None):
    #
    #     if batch_size is None:
    #         batch_size = self.__batch_size
    #
    #     train_events = copy.deepcopy(self.__events)
    #     for i, j in self.pairs():
    #         train_events[i][j] = [t for t in self.__events[i][j] if t <= last_time]
    #
    #     return Events(train_events, batch_size=batch_size)
    #
    # def get_test_events(self, init_time, batch_size=None):
    #
    #     if batch_size is None:
    #         batch_size = self.__batch_size
    #
    #     test_events = copy.deepcopy(self.__events)
    #     for i, j in self.pairs():
    #         test_events[i][j] = [t for t in self.__events[i][j] if t > init_time]
    #
    #     return Events(test_events, batch_size=batch_size)
    #
    # def get_subevents(self, init_time, last_time):
    #
    #     subevents = copy.deepcopy(self.__events)
    #     for i, j in self.pairs():
    #         subevents[i][j] = [t for t in self.__events[i][j] if init_time < t <= last_time]
    #
    #     return Events(subevents)
    #
    # def get_validation_events(self, num):
    #
    #     count = 0
    #     chosen_indices = [[] for _ in range(self.__nodes_num-1)]
    #     while count < num:
    #
    #         node1, node2 = np.random.randint(low=0, high=self.__nodes_num, size=(2, ))
    #
    #         if node1 > node2:
    #             temp = node1
    #             node1 = node2
    #             node2 = temp
    #
    #         if node1 != node2 and node2 not in chosen_indices[node1]:
    #             chosen_indices[node1].append(node2)
    #             count += 1
    #
    #     remaining_events = copy.deepcopy(self.__events)
    #     validation_events = {i: {j: [] for j in range(i+1, self.__nodes_num)} for i in range(self.__nodes_num-1)}
    #     for node1 in range(self.__nodes_num-1):
    #         for node2 in chosen_indices[node1]:
    #             validation_events[node1][node2].extend(remaining_events[node1][node2])
    #             remaining_events[node1][node2].clear()
    #
    #     return Events(remaining_events, batch_size=self.__batch_size), \
    #            Events(validation_events, batch_size=self.__batch_size)

    def get_freq(self):

        F = np.zeros(shape=(self.__nodes_num, self.__nodes_num), dtype=np.int)

        for i, j in zip(*np.triu_indices(self.__nodes_num, k=1)):
            F[i, j] = len(self.__events[i][j])

        return F

    def info(self):

        print(f"Number of nodes: {self.number_of_nodes()}")
        print(f"Number of events: {self.number_of_events()}")
        print(f"The first event time: {self.get_first_event_time()}")
        print(f"The last event time: {self.get_last_event_time()}")
