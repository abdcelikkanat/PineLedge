from datasets.datasets import Dataset
import torch
import matplotlib.pyplot as plt


class Experiments:

    def __init__(self, dataset: Dataset, set_type="test"):

        self.__dataset = dataset
        self.__set_type = set_type

        self.__num_of_nodes = self.__dataset.get_nodes_num()

        self.__data = self.__dataset.get_train_data() if self.__set_type == "train" else self.__dataset.get_test_data()
        self.__node_pairs = [[i, j] for i in range(self.__num_of_nodes) for j in range(i + 1, self.__num_of_nodes)]

        self.__samples, self.__labels = self.__construct_samples()

    def get_samples(self):

        return self.__samples

    def get_labels(self):

        return self.__labels

    def sample_neg_instances(self, num_of_samples, num_of_intervals=10, p=1.0):

        init_time = self.__dataset.get_init_time(set_type=self.__set_type)
        last_time = self.__dataset.get_last_time(set_type=self.__set_type)

        interval_bounds = torch.linspace(init_time, last_time, num_of_intervals)
        num_of_intervals = len(interval_bounds) - 1

        # occurrence_mat = torch.zeros(size=(num_of_nodes, num_of_nodes), dtype=torch.bool)
        # count_mat = torch.zeros(size=(1, num_of_nodes, num_of_nodes), dtype=torch.int)
        #
        # tri_idx = torch.triu_indices(num_of_nodes, num_of_nodes, offset=1)
        # for b in range(num_of_intervals):
        #     occurrence_mat[b, tri_idx[0], tri_idx[1]] = True


        # avg_event_num_per_interval = self.__dataset.get_num_of_events(set_type=self.__set_type) / float(len(node_pairs)) / num_of_intervals
        # print("Neg instance")
        #all_possible_pairs = []
        all_possible_pairs = [[None, None, None] for _ in self.__node_pairs for _ in range(num_of_intervals)]
        # print("Neg instance2")
        current_idx = 0
        for node_pair, events in zip(self.__node_pairs, self.__data):
            idx_list, counts = torch.unique(torch.bucketize(
                events, boundaries=interval_bounds[1:-1], right=True
            ), return_counts=True, sorted=True)

            # if node_pair[0] == 2 and node_pair[1] == 3:
            #
            #     s = torch.bucketize(
            #         events, boundaries=interval_bounds[1:-1], right=True
            #     )
            #     plt.figure()
            #     plt.hist([val.item() for val in s], bins=num_of_intervals)
            #     plt.title("xxxx")
            #     plt.show()

            all_counts = torch.zeros(size=(num_of_intervals, ), dtype=torch.int)
            all_counts[idx_list] = counts.type(torch.int)

            # all_possible_pairs.extend([[node_pair[0], node_pair[1], interval_id] for interval_id in range(num_of_intervals) if all_counts[interval_id] < 1]) #p * avg_event_num_per_interval])
            for interval_id in range(num_of_intervals):
                if all_counts[interval_id] < 1:
                    all_possible_pairs[current_idx] = [node_pair[0], node_pair[1], interval_id]
                    current_idx += 1
        # print("Neg instance3")
        all_possible_pairs = all_possible_pairs[:current_idx]
        # print("Neg instance4")
        # Sample node pairs
        chosen_indices = torch.randint(len(all_possible_pairs), (num_of_samples, ))
        chosen_pairs = torch.as_tensor(all_possible_pairs)[chosen_indices]

        del all_possible_pairs

        # Sample events
        samples = torch.rand(size=(num_of_samples, ), )

        # Map the sampled event times to correct intervals
        interval_size = interval_bounds[1] - interval_bounds[0]
        # print("Neg instance5")
        # print("Neg instance6")
        neg_instances = [[x[0][0], x[0][1], interval_size * x[1] + interval_bounds[x[0][2]]] for x in zip(chosen_pairs, samples)]

        # print("Neg instance7")
        # neg_instances = list(map(
        #     lambda x: [x[0][0], x[0][1], interval_size * x[1] + interval_bounds[x[0][2]]], zip(chosen_pairs, samples)
        # ))

        # plt.figure()
        # z = [s[2].item() for s in chosen_pairs if s[0] == 0 and s[1] == 1]
        # plt.hist(x=z, bins=num_of_intervals, )
        # plt.title("1")
        # plt.show()
        # plt.figure()
        # z = torch.bucketize(torch.as_tensor([s[2].item() for s in neg_instances if s[0] == 0 and s[1] == 1]), boundaries=interval_bounds[1:-1], right=True)
        # plt.hist(x=[zz.item() for zz in z], bins=num_of_intervals, )
        # plt.title("2")
        # plt.show()

        return neg_instances

        # m = torch.sum(count_mat) / float(num_of_intervals * num_of_nodes * (num_of_nodes-1) / 2.0)
        # for b in range(num_of_intervals):
        #     occurrence_mat[b, count_mat[b, :, :] > 0.5*m] = False
        #
        # all_possible_pairs = []
        # for b in range(num_of_intervals):
        #     row_idx, col_idx = torch.nonzero(occurrence_mat[b, :, :], as_tuple=True)
        #     event_times = torch.rand(size=(len(row_idx), )) * (interval_bounds[b+1] - interval_bounds[b]) + interval_bounds[b]
        #     all_possible_pairs.extend([[i, j, e] for i, j, e in zip(row_idx, col_idx, event_times)])
        #
        # chosen_indices = torch.randperm(len(all_possible_pairs))[:sample_size]
        #
        # return [all_possible_pairs[idx] for idx in chosen_indices]

    def __construct_samples(self):

        # print("Sample construction!")
        pos_samples = [[node_pair[0], node_pair[1], e] for node_pair, node_pair_events in zip(self.__node_pairs, self.__data) for e in node_pair_events]
        # for node_pair, node_pair_events in self.__dataset:
        #     for e in node_pair_events:
        #         pos_samples.append([node_pair[0], node_pair[1], e])

        # neg_samples = []
        # for node_pair, node_pair_events in self.__dataset:
        #     non_events = torch.rand(size=(len(node_pair_events), ))*(self.__last_time-self.__init_time)+self.__init_time
        #     for e in non_events:
        #         neg_samples.append([node_pair[0], node_pair[1], e])
        neg_samples = self.sample_neg_instances(num_of_samples=len(pos_samples), num_of_intervals=20)

        # plt.figure()
        # pair_events = [sample[2].item() for sample in neg_samples if sample[0] == 0 and sample[1] == 1]
        # plt.hist(pair_events, bins=torch.linspace(0, 1.0, 50).detach().numpy())
        # plt.show()

        samples = pos_samples + neg_samples
        labels = [1] * len(pos_samples) + [0] * len(neg_samples)

        return samples, labels

    def get_freq_map(self, set_type="train"):

        #nodes = set([node.item() for node_pair, _ in self.__dataset for node in node_pair])
        # num_of_nodes = self.__dataset.get_nodes_num()  # len(nodes)
        # node_pairs = [[i, j] for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)]

        data = self.__dataset.get_train_data() if set_type == "train" else self.__dataset.get_test_data()

        M = torch.zeros(size=(self.__num_of_nodes, self.__num_of_nodes), dtype=torch.int)

        for node_pair, node_pair_events in zip(self.__node_pairs, data):
            M[node_pair[0], node_pair[1]] += len(node_pair_events)

        M = M / M.sum()

        # M = M + M.t()

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
