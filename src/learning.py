import sys
import math
import torch
from src.base import BaseModel
from torch_sparse import spspmm
import time
import utils


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data, nodes_num, bins_num, dim, last_time: float, approach="nhpp",
                 prior_k: int = 4, prior_lambda: float = 1.0,
                 node_pairs_mask: torch.Tensor = None,
                 learning_rate: float = 0.1, batch_size: int = None, epochs_num: int = 100,
                 steps_per_epoch=10, device: torch.device = None, verbose: bool = False, seed: int = 0):

        assert bins_num > 2, print("Number of bins must be greater than 2!")

        super(LearningModel, self).__init__(
            x0=torch.nn.Parameter(2 * torch.rand(size=(nodes_num, dim), device=device) - 1, requires_grad=False),
            v=torch.nn.Parameter(2 * torch.rand(size=(bins_num, nodes_num, dim), device=device) - 1, requires_grad=False),
            beta=torch.nn.Parameter(0 * torch.rand(size=(nodes_num, ), device=device), requires_grad=False),
            bins_num=bins_num,
            last_time=last_time,
            prior_lambda=prior_lambda,
            prior_sigma=torch.nn.Parameter(
                (2.0 / bins_num) * torch.rand(size=(1,), device=device) + (1./bins_num), requires_grad=False
            ),
            prior_B_x0_c=torch.nn.Parameter(torch.ones(size=(1, 1), device=device), requires_grad=False),
            prior_B_sigma=torch.nn.Parameter(
                (1 - (2.0 / bins_num)) * torch.rand(size=(1,), device=device) + (1./bins_num), requires_grad=False
            ),
            prior_C_Q=torch.nn.Parameter(torch.rand(size=(nodes_num, prior_k), device=device), requires_grad=False),
            node_pairs_mask=node_pairs_mask,
            device=device,
            verbose=verbose,
            seed=seed
        )

        self.__data = data
        self.__approach = approach

        self.__learning_procedure = "seq"
        self.__learning_rate = learning_rate
        self.__epochs_num = epochs_num
        self.__steps_per_epoch = steps_per_epoch

        self.__optimizer = None

        self.__verbose = verbose
        self.__device = device

        # Order matters for sequential learning
        self.__learning_param_names = [["x0", "v", ], ["reg_params"], ["beta"]]
        self.__learning_param_epoch_weights = [1, 1, 1]

        self.__add_prior = False  # Do not change

        self.__batch_size = self.get_number_of_nodes() if batch_size is None else batch_size
        self.__events_pairs = torch.as_tensor(self.__data[0], dtype=torch.int, device=self.__device)
        self.__events = self.__data[1]
        # self.__all_lengths = torch.as_tensor(list(map(len, self.__events)), dtype=torch.int, device=self.__device)
        # self.__all_events = torch.as_tensor([e for events in self.__events for e in events], dtype=torch.float, device=self.__device)
        # self.__all_pairs = torch.repeat_interleave(self.__events_pairs, self.__all_lengths, dim=0)
        self.__sampling_weights = torch.ones(self.get_number_of_nodes())
        # self.__sparse_row = (self.__all_pairs[:, 0] * self.get_number_of_nodes())+ self.__all_pairs[:, 1]

        self.__loss = []

        if verbose:
            print("+ Pre-computation process has started...")
            init_time = time.time()
        self.__pair_events = [[[] for _ in range(self._bins_num)] for _ in utils.pair_iter(n=self._nodes_num)]
        print(sys.getsizeof(self.__pair_events), len(self.__pair_events))
        self.__events_count = torch.as_tensor(
            [[0 for _ in range(self._bins_num)] for _ in utils.pair_iter(n=self._nodes_num)]
        )
        self.__alpha1 = torch.as_tensor(
            [[0 for _ in range(self._bins_num)] for _ in utils.pair_iter(n=self._nodes_num)]
        )
        self.__alpha2 = torch.as_tensor(
            [[0 for _ in range(self._bins_num)] for _ in utils.pair_iter(n=self._nodes_num)]
        )
        for pair, events in zip(self.__events_pairs, self.__events):
            flatIdx = utils.pairIdx2flatIdx(i=pair[0], j=pair[1], n=self.get_number_of_nodes())
            batch_idx = torch.div(torch.as_tensor(events), self._bin_width, rounding_mode="floor").type(torch.int)
            batch_idx[batch_idx == self._bins_num] = self._bins_num - 1
            for e, b in zip(events, batch_idx):
                self.__pair_events[flatIdx][b].append(e)

            for b in range(self._bins_num):
                self.__events_count[flatIdx][b] = len(self.__pair_events[flatIdx][b])
                self.__alpha1[flatIdx][b] = torch.sum(utils.remainder(
                    x=torch.as_tensor(self.__pair_events[flatIdx][b]), y=self._bin_width
                ))
                self.__alpha2[flatIdx][b] = torch.sum(utils.remainder(
                    x=torch.as_tensor(self.__pair_events[flatIdx][b]), y=self._bin_width
                ) ** 2)
        if verbose:
            print(f"\t Done! {time.time()-init_time}")

    def learn(self, learning_type=None, loss_file_path=None):

        learning_type = self.__learning_procedure if learning_type is None else learning_type

        # Initialize optimizer list
        self.__optimizer = []

        if learning_type == "seq":

            # For each parameter group, add an optimizer
            for param_group in self.__learning_param_names:

                # Set the gradients to True
                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": True})

                # Add a new optimizer
                self.__optimizer.append(
                    torch.optim.Adam(self.parameters(), lr=self.__learning_rate)
                )
                # Set the gradients to False
                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": False})

            # Run alternating minimization
            self.__sequential_learning()

            if loss_file_path is not None:
                with open(loss_file_path, 'w') as f:
                    for batch_losses in self.__loss:
                        f.write(f"{' '.join('{:.3f}'.format(loss) for loss in batch_losses)}\n")

        elif learning_type == "alt":

            # For each parameter group, add an optimizer
            for param_group in self.__learning_param_names:

                # Set the gradients to True
                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": True})
                # Add a new optimizer
                self.__optimizer.append(
                    torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.__learning_rate)
                )
                # Set the gradients to False
                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": False})

            # Run alternating minimization
            self.__alternating_learning()

        else:

            raise NotImplementedError("A learning method other than alternation minimization is not implemented!")

    def __sequential_learning(self):

        current_epoch = 0
        current_param_group_idx = 0
        group_epoch_counts = (self.__epochs_num * torch.cumsum(
            torch.as_tensor([0] + self.__learning_param_epoch_weights, device=self._device, dtype=torch.float), dim=0
        ) / sum(self.__learning_param_epoch_weights)).type(torch.int)
        group_epoch_counts = group_epoch_counts[1:] - group_epoch_counts[:-1]

        while current_epoch < self.__epochs_num:

            # Set the gradients to True
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": True})

            # Repeat the optimization of the group parameters given weight times
            for _ in range(group_epoch_counts[current_param_group_idx]):
                self.__train_one_epoch(
                    epoch_num=current_epoch, optimizer=self.__optimizer[current_param_group_idx]
                )
                current_epoch += 1

            # Iterate the parameter group id
            current_param_group_idx += 1

    def __alternating_learning(self):

        current_epoch = 0
        current_param_group_idx = 0
        while current_epoch < self.__epochs_num:

            # Set the gradients to True
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": True})

            # Repeat the optimization of the group parameters given weight times
            for _ in range(self.__learning_param_epoch_weights[current_param_group_idx]):

                self.__train_one_epoch(
                    epoch_num=current_epoch, optimizer=self.__optimizer[current_param_group_idx]
                )
                current_epoch += 1

            # Set the gradients to False
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": False})

            # Iterate the parameter group id
            current_param_group_idx = (current_param_group_idx + 1) % len(self.__learning_param_epoch_weights)

    def __train_one_epoch(self, epoch_num, optimizer):

        init_time = time.time()

        average_batch_loss = 0
        self.__loss.append([])
        for batch_num in range(self.__steps_per_epoch):
            batch_loss = self.__train_one_batch(batch_num)
            self.__loss[-1].append(batch_loss)
            average_batch_loss += batch_loss

        # Get the average epoch loss
        epoch_loss = average_batch_loss / float(self.__steps_per_epoch)

        if not math.isfinite(epoch_loss):
            print(f"Epoch loss is {epoch_loss}, stopping training")
            sys.exit(1)

        if self.__verbose and (epoch_num % 10 == 0 or epoch_num == self.__epochs_num - 1):
            print(f"| Epoch = {epoch_num} | Loss/train: {epoch_loss} | Epoch Elapsed time: {time.time() - init_time}")

        # Set the gradients to 0
        optimizer.zero_grad()

        # Backward pass
        epoch_loss.backward()

        # Perform a step
        optimizer.step()

    def __train_one_batch(self, batch_num):

        self.train()

        sampled_nodes = torch.multinomial(self.__sampling_weights, self.__batch_size, replacement=False)
        sampled_nodes, _ = torch.sort(sampled_nodes, dim=0)
        batch_pairs = torch.combinations(sampled_nodes, r=2).T
        indices = (self._nodes_num-1) * batch_pairs[0] - \
                  torch.div(batch_pairs[0]*(batch_pairs[0]+1), 2, rounding_mode="trunc").type(torch.int) + \
                  (batch_pairs[1]-1)

        # Forward pass
        average_batch_loss = self.forward(
            nodes=sampled_nodes, unique_node_pairs=batch_pairs,
            events_count=torch.index_select(self.__events_count, index=indices, dim=0),
            alpha1=torch.index_select(self.__alpha1, index=indices, dim=0),
            alpha2=torch.index_select(self.__alpha2, index=indices, dim=0),
            batch_num=batch_num
        )

        return average_batch_loss

    def forward(self, nodes: torch.Tensor, unique_node_pairs: torch.Tensor,
                events_count: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor, batch_num: int):

        nll = 0
        if self.__approach == "nhpp":
            nll = nll + self.get_negative_log_likelihood(nodes, unique_node_pairs, events_count, alpha1, alpha2)

        elif self.__approach == "survival":
            pass #nll += self.get_survival_log_likelihood(nodes, event_times, event_node_pairs)

        else:
            raise ValueError("Invalid approach name!")

        # Add prior
        nll = nll + self.get_neg_log_prior(batch_nodes=nodes, batch_num=batch_num)

        return nll

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None, reg_params_grad=None):

        if beta_grad is not None:
            self._beta.requires_grad = beta_grad

        if x0_grad is not None:
            self._x0.requires_grad = x0_grad

        if v_grad is not None:
            self._v.requires_grad = v_grad

        if reg_params_grad is not None:

            # Set the gradients of the prior function
            for name, param in self.named_parameters():
                if '_prior' in name:
                    param.requires_grad = reg_params_grad

    def get_hyperparameters(self):

        params = dict()

        params['_nodes_num'] = self._nodes_num
        params['_dim'] = self._dim
        params['_seed'] = self._seed

        for name, param in self.named_parameters():
            # if param.requires_grad:
            params[name.replace(self.__class__.__name__+'_', '')] = param

        params['_prior_lambda'] = self._prior_lambda
        params['_prior_sigma'] = self._prior_sigma

        return params
