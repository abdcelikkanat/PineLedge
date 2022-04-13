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

    def __init__(self, events=None, batch_size=2, seed=0):

        super().__init__()

        self.__nodes_num = None
        self.__nodes = None
        self.__events = None

        # Batch size
        self.__batch_size = batch_size

        # A dictionary mapping index to node pairs
        self.__index2pair = None

        # Set the seed value
        self.__set_seed(seed=seed)

        # if events is not None, initialize the class with a given events
        if events is not None:
            self.__initialize(events, len(events[0].keys())+1)

    def __getitem__(self, idx):

        nodelist = linearIdx2matIdx(idx, n=self.__nodes_num, k=self.__batch_size)

        events = []
        for idx1 in range(self.__batch_size):
            for idx2 in range(idx1+1, self.__batch_size):
                u = nodelist[idx1]
                v = nodelist[idx2]
                events.append(([u, v], self.__events[u][v]))

        return events

    def __len__(self):
        return fac(self.__nodes_num) // fac(self.__batch_size) // fac(self.__nodes_num - self.__batch_size)

    def __set_seed(self, seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def read(self, file_path):

        with open(file_path, 'rb') as f:
            data = pkl.load(f)
        print(data.keys())
        self.__initialize(
            events=data["events"], nodes_num=data.get("n", len(data["events"][0].keys()) + 1)
        )

    def __initialize(self, events, nodes_num):
        # Events are a dictionary of a dictionary storing a list of events
        self.__events = events
        # Get the number of nodes
        self.__nodes_num = nodes_num
        # The nodes are integer values
        self.__nodes = [node for node in range(self.__nodes_num)]
        # A dictionary mapping index to node pairs
        self.__index2pair = {idx: pair for idx, pair in enumerate(self.pairs())}

    def write(self, file_path):

        data = {'events': self.__events, 'n': self.__nodes_num}

        with open(file_path, 'wb') as f:
            pkl.dump(data, f)

    def pairs(self):

        for i in range(self.__nodes_num):
            for j in range(i+1, self.__nodes_num):
                yield i, j

    def number_of_nodes(self):

        return self.__nodes_num

    def number_of_events(self):

        return sum(len(self.__events[pair[0]][pair[1]]) for pair in self.pairs())

    def nodes(self):

        return self.__nodes

    def get_data(self):

        return self.__events

    def normalize(self, init_time=0, last_time=None):

        first_event_time = +1e6
        last_event_time = -1e6

        for i, j in self.pairs():

            events = self.__events[i][j]
            if len(events) > 0:
                min_value, max_value = min(events), max(events)

                if max_value > last_event_time:
                    last_event_time = max_value
                if min_value < first_event_time:
                    first_event_time = min_value

        # Set the last time if not set
        last_time = 1.0 if last_time is None else last_time

        a = (last_time - init_time) / (last_event_time - first_event_time)
        b = init_time - a * first_event_time
        for i, j in self.pairs():
            for idx in range(len(self.__events[i][j])):
                self.__events[i][j][idx] = a * self.__events[i][j][idx] + b

    def get_first_event_time(self):

        min_value = +1e6
        for i, j in self.pairs():
            events = self.__events[i][j]
            if len(events) > 0:
                temp = min(events)
                if min_value > temp:
                    min_value = temp

        return min_value

    def get_last_event_time(self):

        max_value = -1e6
        for i, j in self.pairs():
            events = self.__events[i][j]
            if len(events) > 0:
                temp = max(events)
                if max_value < temp:
                    max_value = temp

        return max_value

    def get_train_events(self, last_time, batch_size=None):

        if batch_size is None:
            batch_size = self.__batch_size

        train_events = copy.deepcopy(self.__events)
        for i, j in self.pairs():
            train_events[i][j] = [t for t in self.__events[i][j] if t <= last_time]

        return Events(train_events, batch_size=batch_size)

    def get_test_events(self, init_time, batch_size=None):

        if batch_size is None:
            batch_size = self.__batch_size

        test_events = copy.deepcopy(self.__events)
        for i, j in self.pairs():
            test_events[i][j] = [t for t in self.__events[i][j] if t > init_time]

        return Events(test_events, batch_size=batch_size)

    def get_subevents(self, init_time, last_time):

        subevents = copy.deepcopy(self.__events)
        for i, j in self.pairs():
            subevents[i][j] = [t for t in self.__events[i][j] if init_time < t <= last_time]

        return Events(subevents)

    def get_validation_events(self, num=None, p=None):

        assert (num is None and p is not None) or (num is not None and p is None), \
            "Number or the percent value must be non-empty!"

        if num is None:
            num = int(self.__nodes_num * (self.__nodes_num - 1) / 2 * p)

        count = 0
        chosen_indices = [[] for _ in range(self.__nodes_num-1)]
        while count < num:

            node1, node2 = np.random.randint(low=0, high=self.__nodes_num, size=(2, ))

            if node1 > node2:
                temp = node1
                node1 = node2
                node2 = temp

            if node1 != node2 and node2 not in chosen_indices[node1]:
                chosen_indices[node1].append(node2)
                count += 1

        remaining_events = copy.deepcopy(self.__events)
        validation_events = {i: {j: [] for j in range(i+1, self.__nodes_num)} for i in range(self.__nodes_num-1)}
        for node1 in range(self.__nodes_num-1):
            for node2 in chosen_indices[node1]:
                validation_events[node1][node2].extend(remaining_events[node1][node2])
                remaining_events[node1][node2].clear()

        return Events(remaining_events, batch_size=self.__batch_size), \
               Events(validation_events, batch_size=self.__batch_size)

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
