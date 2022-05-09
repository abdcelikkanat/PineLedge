import os
import math
import random
import torch
import numpy as np
import pickle as pkl
import copy
import matplotlib.pyplot as plt
from utils import *
import networkx as nx


class Events:

    def __init__(self, path: str = None, data: tuple = None, seed=0):

        self.__events = None
        self.__pairs = None
        self.__nodes_num = None
        self.__nodes = None

        assert path is None or data is None, "Path and data parameter cannot be set at the same time!"

        if path is not None:
            self.read(path)

        if data is not None:
            self.__events = data[0]
            self.__pairs = data[1]

        self.__initialize()


        # Set the seed value
        self.__set_seed(seed=seed)

    def __set_seed(self, seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def read(self, path):

        with open(os.path.join(path, 'events.pkl'), 'rb') as f:
            self.__events = list(pkl.load(f))

        with open(os.path.join(path, 'pairs.pkl'), 'rb') as f:
            self.__pairs = list(pkl.load(f))

        # Make sure that list elements are int
        self.__pairs = np.asarray(self.__pairs, dtype=np.int).tolist()

    def __initialize(self):

        # Set the number of nodes
        self.__nodes_num = len(np.unique(self.__pairs))

        # Set the nodes
        self.__nodes = [node for node in range(self.__nodes_num)]

    def write(self, folder_path):

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, 'events.pkl'), 'wb') as f:
            pkl.dump(self.__events, f)

        with open(os.path.join(folder_path, 'pairs.pkl'), 'wb') as f:
            pkl.dump(self.__pairs, f)

    def __getitem__(self, item):

        if type(item) is int:

            return self.__pairs[item], self.__events[item]

        elif type(item) is tuple or type(item) is list:

            try:
                idx = self.__pairs.index(list(item))
                return self.__pairs[idx], self.__events[idx]

            except ValueError:
                return item, []

        else:

            raise ValueError("Invalid input type!")

    def number_of_nodes(self):

        return self.__nodes_num

    def number_of_event_pairs(self):

        return len(self.__events)

    def number_of_total_events(self):

        return sum(len(events) for events in self.__events)

    def get_min_event_time(self):

        return min([min(pair_events) for pair_events in self.__events])

    def get_max_event_time(self):

        return max([max(pair_events) for pair_events in self.__events])

    def get_nodes(self):

        return self.__nodes

    def get_events(self):

        return self.__events

    def get_pairs(self):

        return self.__pairs

    def normalize(self, init_time=0.0, last_time=1.0):

        min_event_time = self.get_min_event_time()
        max_event_time = self.get_max_event_time()

        for i in range(len(self.__events)):
            for j in range(len(self.__events[i])):
                # Scale it to between 0 and 1
                self.__events[i][j] = (self.__events[i][j] - min_event_time) / (max_event_time - min_event_time)
                # Scale it to between init_time and last_time
                self.__events[i][j] = (last_time - init_time) * self.__events[i][j] - init_time

    def split_events_in_time(self, split_time):

        train_events = []
        train_pairs = []
        test_events = []
        test_pairs = []

        for idx, events in enumerate(self.__events):

            l_list, u_list = [], []
            for e in events:
                if e < split_time:
                    l_list.append(e)
                else:
                    u_list.append(e)

            if len(l_list):
                train_events.append(l_list)
                train_pairs.append(self.__pairs[idx])

            if len(u_list):
                test_events.append(u_list)
                test_pairs.append(self.__pairs[idx])

        te = Events(data=(train_events, train_pairs))

        g = nx.Graph()
        g.add_nodes_from(list(range(self.number_of_nodes())))
        g.add_edges_from(te.get_pairs())

        test_pos_samples = list(g.edges())
        non_edges = list(nx.non_edges(g))

        if len((non_edges)) < len(test_pos_samples):
            raise ValueError("There is no enough number of links containing no event!")

        non_edges = np.asarray(non_edges)
        np.random.shuffle(non_edges)
        test_neg_samples = non_edges[:len(test_pos_samples)]
        test_neg_samples.sort(axis=1)
        test_neg_samples = test_neg_samples.tolist()

        return te, test_pos_samples, test_neg_samples

    # def split_events_in_time(self, split_time):
    #
    #     train_events = []
    #     train_pairs = []
    #     test_events = []
    #     test_pairs = []
    #
    #     for idx, events in enumerate(self.__events):
    #
    #         l_list, u_list = [], []
    #         for e in events:
    #             if e < split_time:
    #                 l_list.append(e)
    #             else:
    #                 u_list.append(e)
    #
    #         if len(l_list):
    #             train_events.append(l_list)
    #             train_pairs.append(self.__pairs[idx])
    #
    #         if len(u_list):
    #             test_events.append(u_list)
    #             test_pairs.append(self.__pairs[idx])
    #
    #     return Events(data=(train_events, train_pairs)), Events(data=(test_events, test_pairs))

    def get_subevents(self, init_time, last_time):

        _, temp_events = self.split_events_in_time(init_time)
        subevents, _ = temp_events.split_events_in_time(last_time)

        return subevents

    def remove_events(self, num, connected=True):

        if not connected:
            raise ValueError("It is not implemented for unconnected networks!")

        g = nx.Graph()
        g.add_edges_from(self.__pairs)

        if not nx.is_connected(g):
            raise ValueError("The network is not connected! {}", list(map(len, nx.connected_components(g))))

        # Shuffle the edges
        edges = list(g.edges())
        np.random.shuffle(edges)

        test_pos_samples = []
        idx = 0
        while len(test_pos_samples) < num:

            edge = edges[idx]

            g.remove_edge(u=edge[0], v=edge[1])
            if nx.is_connected(g):
                test_pos_samples.append([edge[0], edge[1]] if edge[0] < edge[1] else [edge[1], edge[0]])
            else:
                g.add_edge(edge[0], edge[1])

            idx += 1

            if idx == len(edges):
                raise ValueError("There is no enough positive samples keeping the network connected!")

        non_edges = np.asarray(list(nx.non_edges(g)))
        neg_samples_idx = np.random.choice(non_edges.shape[0], size=(num, ), replace=False)
        # Negative samples for testing set
        test_neg_samples = non_edges[neg_samples_idx, :]
        test_neg_samples.sort(axis=1)
        test_neg_samples = test_neg_samples.tolist()

        residual_events = []
        residual_pairs = []
        removed_events = []
        removed_pairs = []
        removed_pair_idx = [self.__pairs.index(pair) for pair in test_pos_samples]
        for idx in range(len(self.__events)):
            if idx in removed_pair_idx:
                removed_events.append(self.__events[idx])
                removed_pairs.append(self.__pairs[idx])
            else:
                residual_events.append(self.__events[idx])
                residual_pairs.append(self.__pairs[idx])

        return Events(data=(residual_events, residual_pairs)), test_pos_samples, test_neg_samples

    # def remove_events(self, num):
    #
    #     chosen_indices = np.random.choice(len(self.__events), size=(num,), replace=False)
    #
    #     residual_events = []
    #     residual_pairs = []
    #     removed_events = []
    #     removed_pairs = []
    #
    #     for idx in range(len(self.__events)):
    #         if idx in chosen_indices:
    #             removed_events.append(self.__events[idx])
    #             removed_pairs.append(self.__pairs[idx])
    #         else:
    #             residual_events.append(self.__events[idx])
    #             residual_pairs.append(self.__pairs[idx])
    #
    #     return Events(data=(residual_events, residual_pairs)), Events(data=(removed_events, removed_pairs))

    def construct_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):

        pos_samples, possible_neg_samples = self.__pos_and_pos_neg_samples(
            bins_num=bins_num, subsampling=subsampling, init_time=init_time, last_time=last_time, with_time=with_time
        )

        if with_time:

            time_gen = np.random.default_rng()
            chosen_idx = np.random.choice(len(possible_neg_samples), size=len(pos_samples), replace=True)
            neg_samples = list(map(
                lambda idx: (
                    possible_neg_samples[idx][0],
                    possible_neg_samples[idx][1],
                    (possible_neg_samples[idx][3] - possible_neg_samples[idx][2])*time_gen.random()+possible_neg_samples[idx][2]
                ), chosen_idx
            ))

        else:
            assert len(possible_neg_samples) >= len(pos_samples), "We couldn't find enough possible negative samples!"

            chosen_idx = np.random.choice(len(possible_neg_samples), size=len(pos_samples), replace=False)
            neg_samples = (np.asarray(possible_neg_samples)[chosen_idx]).tolist()

        all_labels = [1] * len(pos_samples) + [0] * len(neg_samples)
        all_samples = pos_samples + neg_samples

        return all_labels, all_samples

    def __pos_and_pos_neg_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):

        all_pos_samples, all_possible_neg_samples = [], []
        if bins_num > 1:

            bounds = np.linspace(init_time, last_time, bins_num + 1)
            for b in range(bins_num):

                pos_samples, possible_neg_samples = self.__pos_and_pos_neg_samples(
                    bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b],
                    last_time=bounds[b + 1]
                )

                all_pos_samples += pos_samples
                all_possible_neg_samples += possible_neg_samples

        else:

            if init_time is None and last_time is None:
                subevents = self
            else:
                subevents = self.get_subevents(init_time=init_time, last_time=last_time)

            # Sample positive instances
            if with_time:
                pos_samples = [(pair[0], pair[1], t) for pair, events in zip(subevents.get_pairs(), subevents.get_events()) for t in events]
            else:
                pos_samples = [(i, j) for i, j in subevents.get_pairs()]

            if subsampling > 0:
                chosen_samples_indices = np.random.choice(len(pos_samples), size=subsampling, replace=False)
                pos_samples = (np.asarray(pos_samples)[chosen_samples_indices]).tolist()

            possible_neg_samples = [(i, j) for i, j in utils.pair_iter(n=self.__nodes_num) if not len(subevents[(i, j)][1])]
            if with_time:
                possible_neg_samples = [(sample[0], sample[1], init_time, last_time) for sample in possible_neg_samples]

            all_pos_samples = pos_samples
            all_possible_neg_samples = possible_neg_samples

        return all_pos_samples, all_possible_neg_samples

    def plot_events(self, nodes: list = None, fig_size: tuple = None, show = True):

        if nodes is None:
            nodes = [0, 1]

        nodes_num = len(nodes)
        nodes = sorted(nodes)

        pair_indices = [[i, j] for i in range(nodes_num) for j in range(i + 1, nodes_num)]
        pairs = [[nodes[i], nodes[j]] for i, j in pair_indices]

        plt.figure(figsize=fig_size if fig_size is not None else (12, 10))

        for pairIdx, pair in enumerate(pairs):
            _, events = self.__getitem__(pair)
            y = len(events) * [pairIdx]
            plt.plot(events, y, 'k.')

        plt.yticks(np.arange(len(pairs)), [f"({pair[0]},{pair[1]})" for pair in pairs])

        plt.xlabel("Timeline")
        plt.ylabel("Node pairs")

        if show:
            plt.show()

        return plt

    def plot_samples(self, labels, samples, fig_size: tuple = None):

        # Plot the events
        plt = self.plot_events(nodes=list(range(self.number_of_nodes())), fig_size= fig_size, show=False)

        # Check if the samples contain event times
        assert len(samples[0]) == 3, "Samples do not contain event times!"

        c = ['r.', 'b.']
        for label, sample in zip(labels, samples):

            plt.plot(sample[2], utils.pairIdx2flatIdx(i=sample[0], j=sample[1], n=self.number_of_nodes()), c[label])

        plt.show()

    def get_freq(self):

        F = np.zeros(shape=(self.__nodes_num, self.__nodes_num), dtype=np.int)

        for i, j in zip(*np.triu_indices(self.__nodes_num, k=1)):
            F[i, j] = len(self.__events[i][j])

        return F

    def info(self):

        print("- Dataset Information -")
        print(f"\tNumber of nodes: {self.number_of_nodes()}")
        print(f"\tNumber of events: {self.number_of_total_events()}")
        p = round(100 * self.number_of_event_pairs()/(0.5 * self.number_of_nodes() * (self.number_of_nodes() - 1)), 2)
        print(f"\tNumber of pairs having at least one event: {self.number_of_event_pairs()} ({p}%)")
        print(f"\tAverage number of events per pair: {self.number_of_total_events() / float(len(self.__pairs))}")
        print(f"\tMin. time: {self.get_min_event_time()}")
        print(f"\tMax. time: {self.get_max_event_time()}")
