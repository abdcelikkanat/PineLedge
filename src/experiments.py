from datasets.datasets import Dataset
import torch
import matplotlib.pyplot as plt


class Experiments:

    def __init__(self, dataset: Dataset, set_type="test", num_of_bounds=100):

        self.__dataset = dataset
        self.__set_type = set_type
        self.__num_of_bounds = num_of_bounds

        self.__num_of_nodes = self.__dataset.get_num_of_nodes()

        self.__node_pairs = [[i, j] for i in range(self.__num_of_nodes) for j in range(i + 1, self.__num_of_nodes)]
        # events = self.__dataset.get_train_data() if self.__set_type == "train" else self.__dataset.get_test_data()
        # self.__data = [torch.as_tensor(events[p[0]][p[1]]) for p in self.__node_pairs]

        self.__data = self.get_events(set_type=self.__set_type)

        self.__samples, self.__labels = self.__construct_samples()

    def get_events(self, set_type):

        if set_type == "train":
            events = self.__dataset.get_train_data()
        elif self.__set_type == "test":
            events = self.__dataset.get_test_data()
        else:
            raise ValueError("Invalid set type!")

        return [torch.as_tensor(events[p[0]][p[1]]) for p in self.__node_pairs]

    def get_samples(self):

        return self.__samples

    def get_labels(self):

        return self.__labels

    def sample_neg_instances(self, num_of_samples, num_of_bounds=10, p=1.0):

        init_time = self.__dataset.get_init_time(set_type=self.__set_type)
        last_time = self.__dataset.get_last_time(set_type=self.__set_type)

        interval_bounds = torch.linspace(init_time, last_time, num_of_bounds)
        num_of_intervals = num_of_bounds - 1

        all_possible_pairs = [[None, None, None] for _ in self.__node_pairs for _ in range(num_of_intervals)]

        current_idx = 0
        for node_pair, events in zip(self.__node_pairs, self.__data):

            idx_list, counts = torch.unique(torch.bucketize(
                events, boundaries=interval_bounds[1:-1], right=True
            ), return_counts=True, sorted=True)

            all_counts = torch.zeros(size=(num_of_intervals, ), dtype=torch.int)
            all_counts[idx_list] = counts.type(torch.int)

            for interval_id in range(num_of_intervals):
                if all_counts[interval_id] < 1:
                    all_possible_pairs[current_idx] = node_pair[0], node_pair[1], interval_id
                    current_idx += 1

        all_possible_pairs = all_possible_pairs[:current_idx]

        # Sample node pairs
        chosen_indices = torch.randint(len(all_possible_pairs), (num_of_samples, ))
        chosen_pairs = torch.as_tensor(all_possible_pairs)[chosen_indices]
        del all_possible_pairs

        # Sample events
        samples = torch.rand(size=(num_of_samples, ), )

        # Map the sampled event times to correct intervals
        interval_size = interval_bounds[1] - interval_bounds[0]
        neg_instances = [[x[0][0], x[0][1], interval_size * x[1] + interval_bounds[x[0][2]]] for x in zip(chosen_pairs, samples)]

        return neg_instances


    def __construct_samples(self):


        pos_samples = [[node_pair[0], node_pair[1], e] for node_pair, node_pair_events in zip(self.__node_pairs, self.__data) for e in node_pair_events]

        neg_samples = self.sample_neg_instances(num_of_samples=len(pos_samples), num_of_bounds=self.__num_of_bounds)


        samples = pos_samples + neg_samples
        labels = [1] * len(pos_samples) + [0] * len(neg_samples)

        return samples, labels

    def get_freq_map(self, set_type="train"):

        #nodes = set([node.item() for node_pair, _ in self.__dataset for node in node_pair])
        # num_of_nodes = self.__dataset.get_nodes_num()  # len(nodes)
        # node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)]

        data = self.get_events(set_type=set_type)

        M = torch.zeros(size=(self.__num_of_nodes, self.__num_of_nodes), dtype=torch.int)

        for node_pair, node_pair_events in zip(self.__node_pairs, data):
            M[node_pair[0], node_pair[1]] += len(node_pair_events)

        M = M / M.sum()

        return M

    def plot(self):

        num_of_pairs = len(self.__dataset)
        print(num_of_pairs)
        num_of_pairs = 4
        fig, axs = plt.subplots(num_of_pairs, 1)
        for p in range(num_of_pairs):
            node_pair, events = self.__dataset[p]
            axs[p].hist(list(map(float, events)), bins=50)
            axs[p].grid(True)
            axs[p].set_xlabel(f"{node_pair}")
            axs[p].set_ylabel(f"{node_pair}")

        # node_pair, events = self.__dataset[0]
        # bins = torch.linspace(self.__init_time, self.__last_time, 50)
        # print(events.tolist())
        # axs[0].hist(events.tolist(), bins=bins, density=True)
        # axs[0].grid(False)
        # axs[0].set_xlabel(f"{node_pair}")
        #
        plt.show()
        #
        # plt.figure()
        # plt.plot(events, torch.ones(len(events)), '.')
        # plt.show()

    def plot_events(self, u, v, bins=50):

        samples, labels = self.get_samples(), self.get_labels()

        uv_pos_samples = [x[2] for x, y in zip(samples, labels) if x[0] == u and x[1] == v and y == 1]
        uv_neg_samples = [x[2] for x, y in zip(samples, labels) if x[0] == u and x[1] == v and y == 0]

        plt.figure()
        plt.plot(uv_pos_samples, [1] * len(uv_pos_samples), 'b.')
        plt.plot(uv_neg_samples, [-1] * len(uv_neg_samples), 'r.')
        plt.show()
        #
        # plt.figure()
        # plt.plot(events, torch.ones(len(events)), '.')
        # plt.show()

    def get_sample_stats(self, set_type="train"):

        samples, labels = self.get_samples(), self.get_labels()
        num_of_nodes = self.__dataset.get_nodes_num()
        # node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)]

        counts = torch.zeros(size=(2, num_of_nodes, num_of_nodes), dtype=torch.int)
        for s, l in zip(samples, labels):
            counts[l, s[0], s[1]] += 1

        print("------")
        print(counts)
        print("------")
