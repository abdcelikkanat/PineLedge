import os
import math
import torch
import numpy as np
import pickle as pkl
from torch.utils.data.dataset import Dataset

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class DatasetLoader(Dataset):

    def __init__(self, file_path, time_normalization=True):

        self.__file_path = file_path
        self.__time_normalization = time_normalization
        self.__node_pairs, self.__events_list_format = self.__load()
        self.__data = list(zip(self.__node_pairs.transpose(0, 1).unsqueeze(2), self.__events_list_format))

    def __load(self):

        with open(self.__file_path, 'rb') as f:

            data = pkl.load(f)
            node_pairs = data["pairs"]
            events_adj_format = data["events"]

        if self.__time_normalization:

            min_time = +1e6
            max_time = -1e6

            for i, j in zip(node_pairs[0], node_pairs[1]):
                for t in events_adj_format[i.item()][j.item()]:
                    if t > max_time:
                        max_time = t.item()
                    if t < min_time:
                        min_time = t.item()

        events_list_format = []
        for i, j in zip(node_pairs[0], node_pairs[1]):
            events_list_format.append(torch.as_tensor(events_adj_format[i.item()][j.item()]))

        return node_pairs, events_list_format

    def __getitem__(self, idx):

        #return self.__data[idx]
        return self.__node_pairs[:, idx], self.__events_list_format[idx]

    def __len__(self):

        return len(self.__events_list_format)
