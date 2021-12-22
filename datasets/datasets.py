import os
import math
import torch
import numpy as np
import pickle as pkl
from torch.utils.data.dataset import Dataset

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Dataset(Dataset):

    def __init__(self, file_path, init_time=0, last_time=None, time_normalization=True):

        super().__init__()

        self.__init_time = init_time
        self.__last_time = last_time

        self.__file_path = file_path
        self.__time_normalization = time_normalization
        self._num_of_nodes, self._init_time, self._last_time, self.__node_pairs, self.__events_list_format, self.__node2group = self.__load()
        self.__data = list(zip(self.__node_pairs.transpose(0, 1).unsqueeze(2), self.__events_list_format))

    def __load(self):

        with open(self.__file_path, 'rb') as f:

            data = pkl.load(f)
            events_adj_format = data["events"]
            num_of_nodes = len(data["events"][0].keys()) + 1
            node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i+1, num_of_nodes)]
            node_pairs = torch.as_tensor(node_pairs).transpose(0, 1)

        min_time = +1e6
        max_time = -1e6

        for i, j in zip(node_pairs[0], node_pairs[1]):
            for t in events_adj_format[i.item()][j.item()]:
                if t > max_time:
                    max_time = t
                if t < min_time:
                    min_time = t

        if self.__time_normalization:

            assert self.__last_time is not None, "For time normalization, init time must be set!"

            for i, j in zip(node_pairs[0], node_pairs[1]):
                for t_idx, t in enumerate(events_adj_format[i.item()][j.item()]):
                    events_adj_format[i.item()][j.item()][t_idx] = (t - self.__init_time) / float(self.__last_time)

            min_time = 0.0
            max_time = 1.0

        events_list_format = []
        for i, j in zip(node_pairs[0], node_pairs[1]):
            events_list_format.append(torch.as_tensor(events_adj_format[i.item()][j.item()]))

        # Get the node groups
        node2group = data.get("node2group", None)

        return num_of_nodes, min_time, max_time, node_pairs, events_list_format, node2group

    def __getitem__(self, idx):

        #return self.__data[idx]
        return self.__node_pairs[:, idx], self.__events_list_format[idx]

    def __len__(self):

        return len(self.__events_list_format)

    def get_num_of_nodes(self):

        return self._num_of_nodes

    def get_init_time(self):

        return self._init_time

    def get_last_time(self):

        return self._last_time

    def get_groups(self):

        if self.__node2group is None:
            return None

        return list(map(self.__node2group.get, range(self._num_of_nodes)))
