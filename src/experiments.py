from src.events import Events
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


class ExperimentalDesign:

    def __init__(self, events: Events, seed=0):

        self._seed = seed
        self._events = events

        # Set the seed value
        self._set_seed(seed=seed)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def set_events(self, events: Events):

        self._events = events

    def construct_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):

        pos_samples, possible_neg_samples = self._pos_and_pos_neg_samples(
            bins_num=bins_num, subsampling=subsampling, init_time=init_time, last_time=last_time, with_time=with_time
        )

        time_gen = np.random.default_rng()
        chosen_idx = np.random.choice(len(possible_neg_samples), size=len(pos_samples), replace=True)
        neg_samples = list(map(
            lambda idx: (
                possible_neg_samples[idx][0],
                possible_neg_samples[idx][1],
                (possible_neg_samples[idx][3] - possible_neg_samples[idx][2])*time_gen.random()+possible_neg_samples[idx][2]
            ), chosen_idx
        ))

        all_labels = [1] * len(pos_samples) + [0] * len(neg_samples)
        all_samples = pos_samples + neg_samples

        return all_labels, all_samples

    def _pos_and_pos_neg_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):

        all_pos_samples, all_possible_neg_samples = [], []
        if bins_num > 1:

            bounds = np.linspace(init_time, last_time, bins_num + 1)
            for b in range(bins_num):
                '''
                labels, samples = self.construct_samples(
                    bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b], last_time=bounds[b+1]
                )
                all_labels += labels
                all_samples += samples
                '''
                pos_samples, possible_neg_samples = self._pos_and_pos_neg_samples(
                    bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b],
                    last_time=bounds[b + 1]
                )

                all_pos_samples += pos_samples
                all_possible_neg_samples += possible_neg_samples

        else:

            if init_time is None and last_time is None:
                subevents = self.__events.get_data()
            else:
                subevents = self.__events.get_subevents(init_time=init_time, last_time=last_time).get_data()

            # Sample positive instances
            if with_time:
                pos_samples = [(i, j, t) for i, j in self.__pairs() for t in subevents[i][j]]
            else:
                pos_samples = [(i, j) for i, j in self.__pairs() if len(subevents[i][j]) > 0]

            if subsampling > 0:
                pos_samples = np.random.choice(pos_samples, size=subsampling, replace=False).tolist()

            possible_neg_samples = [(i, j) for i, j in self.__pairs() if len(subevents[i][j]) == 0]
            if with_time:
                # time_list = (last_time - init_time) * np.random.random_sample(len(possible_neg_samples)) + init_time
                # possible_neg_samples = [(sample[0], sample[1], t) for sample, t in zip(possible_neg_samples, time_list)]
                possible_neg_samples = [(sample[0], sample[1], init_time, last_time) for sample in possible_neg_samples]

            all_pos_samples = pos_samples
            all_possible_neg_samples = possible_neg_samples

        return all_pos_samples, all_possible_neg_samples



class Experiments_temp:

    def __init__(self, seed=0):

        self.__events: Events = None
        self.__init_time = None
        self.__last_time = None
        self.__nodes_num = None
        self.__pairs = None

        self.__set_seed(seed=seed)

    def __set_seed(self, seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def set_events(self, events: Events):

        self.__events = events
        self.__nodes_num = events.number_of_nodes()
        self.__pairs = events.pairs

    def __pos_and_pos_neg_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):

        all_pos_samples, all_possible_neg_samples = [], []
        if bins_num > 1:

            bounds = np.linspace(init_time, last_time, bins_num+1)
            for b in range(bins_num):
                '''
                labels, samples = self.construct_samples(
                    bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b], last_time=bounds[b+1]
                )
                all_labels += labels
                all_samples += samples
                '''
                pos_samples, possible_neg_samples = self.__pos_and_pos_neg_samples(
                    bins_num=1, subsampling=subsampling, with_time=with_time, init_time=bounds[b],
                    last_time=bounds[b + 1]
                )

                all_pos_samples += pos_samples
                all_possible_neg_samples += possible_neg_samples

        else:

            if init_time is None and last_time is None:
                subevents = self.__events.get_data()
            else:
                subevents = self.__events.get_subevents(init_time=init_time, last_time=last_time).get_data()

            # Sample positive instances
            if with_time:
                pos_samples = [(i, j, t) for i, j in self.__pairs() for t in subevents[i][j]]
            else:
                pos_samples = [(i, j) for i, j in self.__pairs() if len(subevents[i][j]) > 0]

            if subsampling > 0:
                pos_samples = np.random.choice(pos_samples, size=subsampling, replace=False).tolist()

            '''
            # Sample negative instances as much as the positive instances
            possible_neg_samples = np.asarray([(i, j) for i, j in self.__pairs() if len(subevents[i][j]) == 0])
            
            if with_time:
                time_list = (np.random.rand(len(possible_neg_samples)) * (last_time - init_time) + init_time).tolist()
                print(len(possible_neg_samples))
                chosen_idx = np.random.choice(len(possible_neg_samples), size=len(pos_samples), replace=True)
                neg_samples = [(possible_neg_samples[idx][0], possible_neg_samples[idx][1], t) for idx, t in zip(chosen_idx, time_list)]
            else:
                chosen_idx = np.random.choice(len(possible_neg_samples), size=len(pos_samples), replace=False)
                neg_samples = possible_neg_samples[chosen_idx].tolist()

            all_labels = [1]*len(pos_samples) + [0]*len(neg_samples)
            all_samples = pos_samples + neg_samples
            '''

            possible_neg_samples = [(i, j) for i, j in self.__pairs() if len(subevents[i][j]) == 0]
            if with_time:
                # time_list = (last_time - init_time) * np.random.random_sample(len(possible_neg_samples)) + init_time
                # possible_neg_samples = [(sample[0], sample[1], t) for sample, t in zip(possible_neg_samples, time_list)]
                possible_neg_samples = [(sample[0], sample[1], init_time, last_time) for sample in possible_neg_samples]

            all_pos_samples = pos_samples
            all_possible_neg_samples = possible_neg_samples

        return all_pos_samples, all_possible_neg_samples

    def construct_samples(self, bins_num=1, subsampling=0, init_time=None, last_time=None, with_time=False):

        assert self.__events is not None, "Events are not set!"

        pos_samples, possible_neg_samples = self.__pos_and_pos_neg_samples(
            bins_num=bins_num, subsampling=subsampling, init_time=init_time, last_time=last_time, with_time=with_time
        )

        time_gen = np.random.default_rng()
        chosen_idx = np.random.choice(len(possible_neg_samples), size=len(pos_samples), replace=True)
        neg_samples = list(map(
            lambda idx: (
                possible_neg_samples[idx][0],
                possible_neg_samples[idx][1],
                (possible_neg_samples[idx][3] - possible_neg_samples[idx][2])*time_gen.random()+possible_neg_samples[idx][2]
            ), chosen_idx
        ))

        all_labels = [1] * len(pos_samples) + [0] * len(neg_samples)
        all_samples = pos_samples + neg_samples

        return all_labels, all_samples

    def get_freq_map(self, events_data=None):

        if events_data is None:
            events_data = self.__events.get_data()

        M = torch.zeros(size=(self.__nodes_num, self.__nodes_num), dtype=torch.int)
        for i, j in self.__pairs():
            M[i, j] = len(events_data[i][j])

        M = M / M.sum()

        return M

    def plot_samples(self, labels, samples, figsize=None):

        assert len(samples[0]) == 3, "Each sample tuple must be in the form of (i,j,t)."

        # Red and black indicate negative and black samples.
        colors = ['r', 'k']

        pair2idx = {(i, j): i*self.__nodes_num+j-(i+1)*(i+2)/2 for i in range(self.__nodes_num) for j in range(i+1, self.__nodes_num)}
        plt.figure(figsize=figsize)
        for label, sample in zip(labels, samples):
            if colors[label] == 'k':
                plt.plot(sample[2], pair2idx[(sample[0], sample[1])], f".{colors[label]}")

        plt.yticks(
            [pair2idx[(i, j)] for i in range(self.__nodes_num) for j in range(i+1, self.__nodes_num)],
            [f"({i}, {j})" for i in range(self.__nodes_num) for j in range(i+1, self.__nodes_num)]
        )
        plt.xticks(np.arange(0, 1.1, 0.1))

        plt.show()

    # def construct_samples(self):
    #
    #     pos_samples = [[node_pair[0], node_pair[1], e] for node_pair, node_pair_events in zip(self.__node_pairs, self.__data) for e in node_pair_events]
    #
    #     neg_samples = self.sample_neg_instances(num_of_samples=len(pos_samples), num_of_bounds=self.__num_of_bounds)
    #
    #     samples = pos_samples + neg_samples
    #     labels = [1] * len(pos_samples) + [0] * len(neg_samples)
    #
    #     return samples, labels
    #
    # def get_events(self, set_type):
    #
    #     if set_type == "train":
    #         events = self.__dataset.get_train_data()
    #     elif self.__set_type == "test":
    #         events = self.__dataset.get_test_data()
    #     else:
    #         raise ValueError("Invalid set type!")
    #
    #     return [torch.as_tensor(events[p[0]][p[1]]) for p in self.__node_pairs]
    #
    # # def get_samples(self):
    # #
    # #     return self.__samples
    # #
    # # def get_labels(self):
    # #
    # #     return self.__labels
    #
    # def sample_neg_instances(self, num_of_samples, num_of_bounds=10, p=1.0):
    #
    #     init_time = self.__dataset.get_init_time(set_type=self.__set_type)
    #     last_time = self.__dataset.get_last_time(set_type=self.__set_type)
    #
    #     interval_bounds = torch.linspace(init_time, last_time, num_of_bounds)
    #     num_of_intervals = num_of_bounds - 1
    #
    #     all_possible_pairs = [[None, None, None] for _ in self.__node_pairs for _ in range(num_of_intervals)]
    #
    #     current_idx = 0
    #     for node_pair, events in zip(self.__node_pairs, self.__data):
    #
    #         idx_list, counts = torch.unique(torch.bucketize(
    #             events, boundaries=interval_bounds[1:-1], right=True
    #         ), return_counts=True, sorted=True)
    #
    #         all_counts = torch.zeros(size=(num_of_intervals, ), dtype=torch.int)
    #         all_counts[idx_list] = counts.type(torch.int)
    #
    #         for interval_id in range(num_of_intervals):
    #             if all_counts[interval_id] < 1:
    #                 all_possible_pairs[current_idx] = node_pair[0], node_pair[1], interval_id
    #                 current_idx += 1
    #
    #     all_possible_pairs = all_possible_pairs[:current_idx]
    #
    #     # Sample node pairs
    #     chosen_indices = torch.randint(len(all_possible_pairs), (num_of_samples, ))
    #     chosen_pairs = torch.as_tensor(all_possible_pairs)[chosen_indices]
    #     del all_possible_pairs
    #
    #     # Sample events
    #     samples = torch.rand(size=(num_of_samples, ), )
    #
    #     # Map the sampled event times to correct intervals
    #     interval_size = interval_bounds[1] - interval_bounds[0]
    #     neg_instances = [[x[0][0], x[0][1], interval_size * x[1] + interval_bounds[x[0][2]]] for x in zip(chosen_pairs, samples)]
    #
    #     return neg_instances
    #
    #
    #
    # def get_freq_map(self, set_type="train"):
    #
    #     #nodes = set([node.item() for node_pair, _ in self.__dataset for node in node_pair])
    #     # num_of_nodes = self.__dataset.get_nodes_num()  # len(nodes)
    #     # node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)]
    #
    #     data = self.get_events(set_type=set_type)
    #
    #     M = torch.zeros(size=(self.__num_of_nodes, self.__num_of_nodes), dtype=torch.int)
    #
    #     for node_pair, node_pair_events in zip(self.__node_pairs, data):
    #         M[node_pair[0], node_pair[1]] += len(node_pair_events)
    #
    #     M = M / M.sum()
    #
    #     return M
    #
    # def plot(self):
    #
    #     num_of_pairs = len(self.__dataset)
    #     print(num_of_pairs)
    #     num_of_pairs = 4
    #     fig, axs = plt.subplots(num_of_pairs, 1)
    #     for p in range(num_of_pairs):
    #         node_pair, events = self.__dataset[p]
    #         axs[p].hist(list(map(float, events)), bins=50)
    #         axs[p].grid(True)
    #         axs[p].set_xlabel(f"{node_pair}")
    #         axs[p].set_ylabel(f"{node_pair}")
    #
    #     # node_pair, events = self.__dataset[0]
    #     # bins = torch.linspace(self.__init_time, self.__last_time, 50)
    #     # print(events.tolist())
    #     # axs[0].hist(events.tolist(), bins=bins, density=True)
    #     # axs[0].grid(False)
    #     # axs[0].set_xlabel(f"{node_pair}")
    #     #
    #     plt.show()
    #     #
    #     # plt.figure()
    #     # plt.plot(events, torch.ones(len(events)), '.')
    #     # plt.show()
    #
    # def plot_events(self, u, v, bins=50):
    #
    #     samples, labels = self.get_samples(), self.get_labels()
    #
    #     uv_pos_samples = [x[2] for x, y in zip(samples, labels) if x[0] == u and x[1] == v and y == 1]
    #     uv_neg_samples = [x[2] for x, y in zip(samples, labels) if x[0] == u and x[1] == v and y == 0]
    #
    #     plt.figure()
    #     plt.plot(uv_pos_samples, [1] * len(uv_pos_samples), 'b.')
    #     plt.plot(uv_neg_samples, [-1] * len(uv_neg_samples), 'r.')
    #     plt.show()
    #     #
    #     # plt.figure()
    #     # plt.plot(events, torch.ones(len(events)), '.')
    #     # plt.show()
    #
    # def get_sample_stats(self, set_type="train"):
    #
    #     samples, labels = self.get_samples(), self.get_labels()
    #     num_of_nodes = self.__dataset.get_nodes_num()
    #     # node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)]
    #
    #     counts = torch.zeros(size=(2, num_of_nodes, num_of_nodes), dtype=torch.int)
    #     for s, l in zip(samples, labels):
    #         counts[l, s[0], s[1]] += 1
    #
    #     print("------")
    #     print(counts)
    #     print("------")


# path = "../datasets/real/fb_forum/fb_forum_events.pkl"
# events = Events()
# events.read(file_path=path)
# events.normalize(0, 1)
# train_events = events.get_train_events(last_time=0.9)
# exp = Experiments()
# exp.set_events(events=train_events, init_time=0, last_time=0.9)
# labels, samples = exp.construct_samples(bins_num=1, subsampling=0, with_time=True)
#
# print(samples)

# class Experiments:
#
#     def __init__(self, dataset: Dataset, set_type="test", num_of_bounds=100):
#
#         self.__dataset = dataset
#         self.__set_type = set_type
#         self.__num_of_bounds = num_of_bounds
#
#         self.__num_of_nodes = self.__dataset.get_num_of_nodes()
#
#         self.__node_pairs = [[i, j] for i in range(self.__num_of_nodes) for j in range(i + 1, self.__num_of_nodes)]
#
#         self.__data = self.get_events(set_type=self.__set_type)
#
#         # self.__samples, self.__labels = self.__construct_samples()
#
#     def construct_samples(self):
#
#         pos_samples = [[node_pair[0], node_pair[1], e] for node_pair, node_pair_events in zip(self.__node_pairs, self.__data) for e in node_pair_events]
#
#         neg_samples = self.sample_neg_instances(num_of_samples=len(pos_samples), num_of_bounds=self.__num_of_bounds)
#
#         samples = pos_samples + neg_samples
#         labels = [1] * len(pos_samples) + [0] * len(neg_samples)
#
#         return samples, labels
#
#     def get_events(self, set_type):
#
#         if set_type == "train":
#             events = self.__dataset.get_train_data()
#         elif self.__set_type == "test":
#             events = self.__dataset.get_test_data()
#         else:
#             raise ValueError("Invalid set type!")
#
#         return [torch.as_tensor(events[p[0]][p[1]]) for p in self.__node_pairs]
#
#     # def get_samples(self):
#     #
#     #     return self.__samples
#     #
#     # def get_labels(self):
#     #
#     #     return self.__labels
#
#     def sample_neg_instances(self, num_of_samples, num_of_bounds=10, p=1.0):
#
#         init_time = self.__dataset.get_init_time(set_type=self.__set_type)
#         last_time = self.__dataset.get_last_time(set_type=self.__set_type)
#
#         interval_bounds = torch.linspace(init_time, last_time, num_of_bounds)
#         num_of_intervals = num_of_bounds - 1
#
#         all_possible_pairs = [[None, None, None] for _ in self.__node_pairs for _ in range(num_of_intervals)]
#
#         current_idx = 0
#         for node_pair, events in zip(self.__node_pairs, self.__data):
#
#             idx_list, counts = torch.unique(torch.bucketize(
#                 events, boundaries=interval_bounds[1:-1], right=True
#             ), return_counts=True, sorted=True)
#
#             all_counts = torch.zeros(size=(num_of_intervals, ), dtype=torch.int)
#             all_counts[idx_list] = counts.type(torch.int)
#
#             for interval_id in range(num_of_intervals):
#                 if all_counts[interval_id] < 1:
#                     all_possible_pairs[current_idx] = node_pair[0], node_pair[1], interval_id
#                     current_idx += 1
#
#         all_possible_pairs = all_possible_pairs[:current_idx]
#
#         # Sample node pairs
#         chosen_indices = torch.randint(len(all_possible_pairs), (num_of_samples, ))
#         chosen_pairs = torch.as_tensor(all_possible_pairs)[chosen_indices]
#         del all_possible_pairs
#
#         # Sample events
#         samples = torch.rand(size=(num_of_samples, ), )
#
#         # Map the sampled event times to correct intervals
#         interval_size = interval_bounds[1] - interval_bounds[0]
#         neg_instances = [[x[0][0], x[0][1], interval_size * x[1] + interval_bounds[x[0][2]]] for x in zip(chosen_pairs, samples)]
#
#         return neg_instances
#
#
#
#     def get_freq_map(self, set_type="train"):
#
#         #nodes = set([node.item() for node_pair, _ in self.__dataset for node in node_pair])
#         # num_of_nodes = self.__dataset.get_nodes_num()  # len(nodes)
#         # node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)]
#
#         data = self.get_events(set_type=set_type)
#
#         M = torch.zeros(size=(self.__num_of_nodes, self.__num_of_nodes), dtype=torch.int)
#
#         for node_pair, node_pair_events in zip(self.__node_pairs, data):
#             M[node_pair[0], node_pair[1]] += len(node_pair_events)
#
#         M = M / M.sum()
#
#         return M
#
#     def plot(self):
#
#         num_of_pairs = len(self.__dataset)
#         print(num_of_pairs)
#         num_of_pairs = 4
#         fig, axs = plt.subplots(num_of_pairs, 1)
#         for p in range(num_of_pairs):
#             node_pair, events = self.__dataset[p]
#             axs[p].hist(list(map(float, events)), bins=50)
#             axs[p].grid(True)
#             axs[p].set_xlabel(f"{node_pair}")
#             axs[p].set_ylabel(f"{node_pair}")
#
#         # node_pair, events = self.__dataset[0]
#         # bins = torch.linspace(self.__init_time, self.__last_time, 50)
#         # print(events.tolist())
#         # axs[0].hist(events.tolist(), bins=bins, density=True)
#         # axs[0].grid(False)
#         # axs[0].set_xlabel(f"{node_pair}")
#         #
#         plt.show()
#         #
#         # plt.figure()
#         # plt.plot(events, torch.ones(len(events)), '.')
#         # plt.show()
#
#     def plot_events(self, u, v, bins=50):
#
#         samples, labels = self.get_samples(), self.get_labels()
#
#         uv_pos_samples = [x[2] for x, y in zip(samples, labels) if x[0] == u and x[1] == v and y == 1]
#         uv_neg_samples = [x[2] for x, y in zip(samples, labels) if x[0] == u and x[1] == v and y == 0]
#
#         plt.figure()
#         plt.plot(uv_pos_samples, [1] * len(uv_pos_samples), 'b.')
#         plt.plot(uv_neg_samples, [-1] * len(uv_neg_samples), 'r.')
#         plt.show()
#         #
#         # plt.figure()
#         # plt.plot(events, torch.ones(len(events)), '.')
#         # plt.show()
#
#     def get_sample_stats(self, set_type="train"):
#
#         samples, labels = self.get_samples(), self.get_labels()
#         num_of_nodes = self.__dataset.get_nodes_num()
#         # node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)]
#
#         counts = torch.zeros(size=(2, num_of_nodes, num_of_nodes), dtype=torch.int)
#         for s, l in zip(samples, labels):
#             counts[l, s[0], s[1]] += 1
#
#         print("------")
#         print(counts)
#         print("------")
