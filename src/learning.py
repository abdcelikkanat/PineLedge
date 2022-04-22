import sys
import math
import torch
from src.base import BaseModel
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from torch_sparse import spspmm
from utils import mean_normalization
import utils
import time


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data, nodes_num, bins_num, dim, last_time: float,
                 learning_rate: float, pw: float = 1.0, batch_size: float = None, epochs_num: int = 100,
                 steps_per_epoch=10, device: torch.device = "cpu", verbose: bool = False, seed: int = 0,
                 approach: str = "nhpp"):

        super(LearningModel, self).__init__(
            x0=torch.nn.Parameter(2 * torch.rand(size=(nodes_num, dim), device=device) - 1, requires_grad=False),
            v=torch.nn.Parameter(2 * torch.rand(size=(bins_num, nodes_num, dim), device=device) - 1, requires_grad=False),
            beta=torch.nn.Parameter(0.0 * torch.ones(size=(1,), device=device), requires_grad=False),
            bins_rwidth=torch.nn.Parameter(torch.zeros(size=(bins_num,), device=device) / float(bins_num), requires_grad=False),
            last_time=last_time,
            device=device,
            verbose=verbose,
            seed=seed
        )

        self._scale_const = 1.0 #1e3
        self.initialize_prior_params()

        self.__data = data

        # Set the prior function
        # self.__neg_log_prior = self.__set_prior( kernels=["rbf",]) # rbf periodic
        self.__approach = approach

        # Set the correction function
        self.__correction_func = None

        self.__learning_procedure = "alt"  #"seq"
        self.__learning_rate = learning_rate
        self.__epochs_num = epochs_num
        self.__steps_per_epoch = steps_per_epoch
        self.__pw = pw

        #self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__learning_rate)
        self.__optimizer = None

        self.__verbose = verbose
        self.__device = device
        self.__writer = SummaryWriter("../experiments/logs/loss")

        # Order matters for sequential learning
        self.__learning_param_names = [ ["x0", "v",], ["reg_params"], ["beta"] ]  # , ["reg_params"] ["bins_rwidth"] "reg_params" "bins_rwidth" ["v", "bins_rwidth"], ["beta"], ["bins_rwidth"]
        self.__learning_param_epoch_weights = [1, 2, 1 ]  # 2

        self.__add_prior = False  # Do not change

        self.__batch_size = self.get_number_of_nodes() if batch_size is None else batch_size
        self.__events_pairs = torch.as_tensor(self.__data[0], dtype=torch.int, device=self.__device)
        self.__events = self.__data[1]
        self.__all_lengths = torch.as_tensor(list(map(len, self.__events)), dtype=torch.int, device=self.__device)
        self.__all_events = torch.as_tensor([e for events in self.__events for e in events], dtype=torch.float, device=self.__device)
        self.__all_pairs = torch.repeat_interleave(self.__events_pairs, self.__all_lengths, dim=0)
        self.__sampling_weights = torch.ones(self.get_number_of_nodes())
        self.__sparse_row = (self.__all_pairs[:, 0]*self.get_number_of_nodes())+self.__all_pairs[:,1]

        self.__softplus = torch.nn.Softplus()

    def learn(self, learning_type=None):

        learning_type = self.__learning_procedure if learning_type is None else learning_type

        # Learns the parameters sequentially
        if learning_type == "seq":

            self.__sequential_learning()

        elif learning_type == "alt":

            self.__optimizer = []
            # Set the gradients to True
            for param_group in self.__learning_param_names:

                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": True})

                self.__optimizer.append(
                    torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.__learning_rate)
                )

                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": False})

            # self.__alternating_learning()
            self.__alternating_learning()

        else:

            raise NotImplementedError("Non-sequential learning not implemented!")

        if self.__writer is not None:
            self.__writer.close()

    def __alternating_learning(self):

        current_epoch = 0
        current_param_group_idx = 0
        while current_epoch < self.__epochs_num:

            # Set the gradients to True
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": True})

            # self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__learning_rate)

            for _ in range(self.__learning_param_epoch_weights[current_param_group_idx]):

                self.__train_one_epoch(epoch=current_epoch, correction_func=self.__correction_func, optimizer=self.__optimizer[current_param_group_idx])
                current_epoch += 1

            # Set the gradients to False
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": False})

            # Iterate the parameter group id
            current_param_group_idx = (current_param_group_idx + 1) % len(self.__learning_param_epoch_weights)

    def __train_one_epoch(self, epoch, optimizer, correction_func=None):

        init_time = time.time()

        average_epoch_loss = 0
        batch_num = 0

        for _ in range(self.__steps_per_epoch):

            self.train()

            sample_indices = torch.multinomial(self.__sampling_weights, self.__batch_size, replacement=False)
            sample_pairs = ((sample_indices*self.get_number_of_nodes()).unsqueeze(1)+sample_indices).reshape(-1).unsqueeze(0)

            c = 1
            indexC, valueC = spspmm(
                indexA=sample_pairs.repeat(2, 1).long(),
                valueA=torch.ones(sample_pairs.shape[1]),
                indexB=torch.cat((self.__sparse_row.unsqueeze(0), torch.arange(self.__all_events.shape[0]).unsqueeze(0)), 0).long(),
                valueB=(self.__all_events + c),
                m=self.get_number_of_nodes()**2,
                k=self.get_number_of_nodes()**2,
                n=self.__all_events.shape[0],
                coalesced=True
            )
            valueC = valueC - c

            unique_indexC0, inverse_indices = torch.unique(indexC[0], return_inverse=True)

            sample_i = torch.div(unique_indexC0, self.get_number_of_nodes(), rounding_mode='floor').unsqueeze(1)
            sample_j = (unique_indexC0 % self.get_number_of_nodes()).unsqueeze(1)

            batch_node_pairs = torch.hstack((sample_i, sample_j)).t()
            batch_times_list = [[] for _ in range(len(unique_indexC0))] # ---> This part will be fixed!
            for i in range(len(valueC)):
                batch_times_list[inverse_indices[i]].append(valueC[i])

            # print(batch_node_pairs)
            # print(self.__events[0])
            # batch_node_pairs, batch_times_list = next(iter(self.__data))
            # batch_node_pairs, batch_times_list = batch_node_pairs.to(self._device), batch_times_list

            # Forward pass
            batch_loss_sum = self.forward(
                time_seq_list=batch_times_list, node_pairs=batch_node_pairs,
            )

            # Store the average batch losses
            batch_events_count = sum(map(len, batch_times_list))
            if batch_events_count > 0:
                average_batch_loss_value = batch_loss_sum.item() / float(batch_events_count)

                if not math.isfinite(average_batch_loss_value):
                    print(f"Batch loss is {average_batch_loss_value}, stopping training")
                    sys.exit(1)

                average_epoch_loss += average_batch_loss_value

            # Set the gradients to 0
            optimizer.zero_grad()

            # Backward pass
            batch_loss_sum.backward()

            # Perform a step
            optimizer.step()

            if correction_func is not None:
                correction_func()

            # Increment batch number
            batch_num += 1

        average_epoch_loss = average_epoch_loss / float(self.__steps_per_epoch)

        if self.__writer is not None:
            self.__writer.add_scalar(tag="Loss/train", scalar_value=average_epoch_loss, global_step=epoch)

        if self.__verbose and (epoch % 10 == 0 or epoch == self.__epochs_num - 1):
            print(f"| Epoch = {epoch} | Loss/train: {average_epoch_loss} | Elapsed time: {time.time() - init_time}")

    def forward(self, time_seq_list, node_pairs):

        if self.__approach == "nhpp":
            nll = self.get_negative_log_likelihood(time_seq_list, node_pairs)
        elif self.__approach == "survival":
            nll = self.get_survival_log_likelihood(time_seq_list, node_pairs)
        else:
            raise ValueError("Invalid approach name!")

        nll += self.__neg_log_prior(nodes=torch.unique(node_pairs))

        return nll

    def initialize_prior_params(self, kernel_names: list = None):

        # Initialize the prior terms
        self.__prior_sigma = torch.nn.Parameter(
            2 * torch.rand(size=(1,)) - 1, requires_grad=False
        )

        # Set the parameters required for the construction of the matrix B
        self.__prior_B_sigma = torch.nn.Parameter(
            2 * torch.rand(size=(1,)) - 1, requires_grad=False
        )

        # Set the noise term for the kernel
        self.__prior_B_noise = torch.nn.Parameter(
            2 * torch.rand(size=(1,)) - 1, requires_grad=False
        )
        # Set the parameters required for the construction of the matrix C
        self.__prior_C_Q_dim = 30
        self.__prior_C_Q = torch.nn.Parameter(
            2.0 * torch.rand(size=(self.__prior_C_Q_dim, self._nodes_num)) - 1.0, requires_grad=False
        )

        # # Set the parameters required for the construction of the matrix D
        # # Set the noise term for the kernel
        # self.__prior_D_noise = torch.nn.Parameter(
        #     (2.0 / self._scale_const) * torch.rand(size=(1,)) - (1.0 / self._scale_const), requires_grad=False
        # )

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
                if '__prior' in name:
                    param.requires_grad = reg_params_grad

    def __get_rbf_kernel(self, time_mat, sigma):

        kernel = torch.exp(-0.5 * torch.div(time_mat**2, sigma**2))

        return kernel

    def __get_B_factor(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor, get_inv: bool = True, cholesky: bool = True):

        time_mat = bin_centers1 - bin_centers2.transpose(1, 2)
        time_mat = time_mat.squeeze(0)

        B_sigma = self.__softplus(self.__prior_B_sigma)
        kernel = torch.exp(-0.5 * torch.div(time_mat**2, B_sigma))

        # Add a noise term
        #kernel = kernel #+ torch.eye(n=kernel.shape[0], m=kernel.shape[1]) * (self.__prior_B_noise**2)

        # B x B upper triangular matrix
        U = torch.linalg.cholesky(kernel).t()

        return U

    def __get_C_factor(self):

        # K x N matrix
        return torch.softmax(self.__prior_C_Q, dim=0)

    def __get_D_factor(self):

        # D x D matrix
        return torch.eye(self.get_dim())

    def __neg_log_prior(self, nodes, cholesky=True):

        # Get the number of bin size
        bin_num = self.get_num_of_bins()

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, bin_num) / 2.

        # B x B matrix
        B_factor = self.__get_B_factor(bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=cholesky)
        # K x N matrix where K is the community size
        C_factor = self.__get_C_factor()
        # D x D matrix
        D_factor = self.__get_D_factor()

        # BKD x BND matrix
        S = torch.kron(B_factor, torch.kron(C_factor, D_factor))

        pw = torch.as_tensor(self.__pw) #torch.sigmoid(torch.as_tensor(self.__pw))
        sigma = torch.sigmoid(torch.as_tensor(self.__prior_sigma))
        final_dim = self.get_number_of_nodes() * self.get_num_of_bins() * self._dim
        reduced_dim = self.get_num_of_bins() * self.__prior_C_Q_dim * self._dim
        inv_sigma_sq = 1.0 / (sigma**2)
        inv_pw_sq = 1.0 / pw  # lambda**2

        # Batching
        B_index = torch.ones(size=(self.get_num_of_bins(),), dtype=torch.int)
        C_index = torch.zeros(size=(self.get_number_of_nodes(),), dtype=torch.int)
        C_index[nodes] = 1
        D_index = torch.ones(size=(self.get_dim(),), dtype=torch.int)
        chosen_index_mask = torch.kron(B_index, torch.kron(C_index, D_index))
        indices = torch.arange(final_dim) * chosen_index_mask

        # Compute the inverse of the covariance matrix by Woodbury matrix identity
        R = torch.eye(reduced_dim) + inv_sigma_sq * (S @ S.t())
        R_inv = torch.cholesky_inverse(torch.linalg.cholesky(R))
        S_batch = torch.index_select(S, dim=1, index=indices)
        K_inv = inv_pw_sq * (inv_sigma_sq*torch.eye(torch.sum(chosen_index_mask)) - inv_sigma_sq * S_batch.t() @ R_inv @ S_batch * inv_sigma_sq)

        # Compute the determinant by matrix determinant lemma
        log_det_kernel = torch.logdet(
            torch.eye(reduced_dim) + inv_sigma_sq * (S @ S.t())
        ) + final_dim * (torch.log(sigma**2) + torch.log(pw**2))

        batch_v = torch.index_select(mean_normalization(self._v), dim=1, index=nodes)
        v_vect = batch_v.flatten()

        p = v_vect.t() @ K_inv @ v_vect

        log_prior_likelihood = -0.5 * (final_dim * math.log(2 * math.pi) + log_det_kernel + p)

        return -log_prior_likelihood.squeeze(0)

    # def __neg_log_prior(self, nodes, cholesky=True):
    #
    #     # Get the number of bin size
    #     bin_num = self.get_num_of_bins()
    #
    #     # Get the bin bounds
    #     bounds = self.get_bins_bounds()
    #
    #     # Get the middle time points of the bins for TxT covariance matrix
    #     middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, bin_num) / 2.
    #
    #     # B x B matrix
    #     B, inv_B = self.__get_B(
    #         bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=cholesky
    #     )
    #
    #     # len(nodes) x len(nodes) matrix
    #     inv_C = self.__get_C(nodes=nodes)
    #
    #     # D x D matrix
    #     inv_D = self.__get_D()
    #
    #     # Compute the product, v^t ( B kron C kron D )^-1 v,  in an efficient way
    #     batch_v = torch.index_select(mean_normalization(self._v), dim=1, index=nodes)
    #     v_vect = utils.vectorize(batch_v).flatten()
    #
    #     # inv_K = (1.0 / self.__pw**2) * torch.kron(torch.kron(inv_B.contiguous(), inv_C.contiguous()), inv_D.contiguous())
    #     # p = v_vect @ inv_K @ v_vect
    #
    #     p = (1.0 / self.__pw**2) * torch.matmul(
    #         v_vect,
    #         utils.vectorize(torch.matmul(
    #             utils.vectorize(torch.matmul(torch.matmul(inv_D.unsqueeze(0), batch_v.transpose(1, 2)),
    #                                          inv_C.transpose(0, 1).unsqueeze(0))).transpose(0, 1),
    #             inv_B.transpose(0, 1)
    #         ))
    #     )
    #
    #     # Compute the log-determinant of the product
    #     final_dim = len(nodes) * self.get_num_of_bins() * self._dim
    #     log_det_kernel = (final_dim / self.get_num_of_bins()) * torch.logdet(B) \
    #                      - (final_dim / self._dim) * torch.logdet(inv_C) \
    #                      - (final_dim / len(nodes)) * torch.logdet(inv_D)
    #
    #     log_prior_likelihood = -0.5 * (final_dim * math.log(2 * math.pi) + log_det_kernel + p)
    #
    #     return -log_prior_likelihood.squeeze(0)

    # def __get_B(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor,
    #             get_inv: bool = True, cholesky: bool = True):
    #
    #     # Compute the inverse of kernel/covariance matrix
    #     time_mat = bin_centers1 - bin_centers2.transpose(1, 2)
    #     time_mat = time_mat.squeeze(0)
    #
    #     kernel = self.__get_rbf_kernel(time_mat=time_mat, sigma=self.__prior_B_sigma)
    #
    #     # Add a noise term
    #     kernel = kernel + torch.eye(n=kernel.shape[0], m=kernel.shape[1]) * (self.__prior_B_noise**2)
    #
    #     # If the inverse of the kernel is not required, return only the kernel matrix
    #     if not get_inv:
    #         return kernel
    #
    #     # Compute the inverse
    #     if cholesky:
    #         inv_kernel = torch.cholesky_inverse(torch.linalg.cholesky(kernel))
    #     else:
    #         inv_kernel = torch.linalg.inv(kernel)
    #
    #     return kernel, inv_kernel

    # def __get_C(self, nodes=None):
    #
    #     # C: N x N matrix
    #     nodes_num = self.get_number_of_nodes() if nodes is None else len(nodes)
    #
    #     Q = self.__prior_C_Q
    #     R = torch.linalg.inv(torch.eye(self.__prior_C_Q.shape[1]) + self.__prior_C_Q.t() @ self.__prior_C_Q)
    #
    #     if nodes is not None:
    #         Q = torch.index_select(self.__prior_C_Q, dim=0, index=nodes)
    #
    #     inv_C = torch.eye(nodes_num) - Q @ R @ Q.t()
    #
    #     return inv_C
    #
    # def __get_D(self):
    #
    #     # D: D x D matrix
    #     inv_D = torch.eye(n=self._dim, m=self._dim) / (1.0 + self.__prior_D_noise**2)
    #
    #     return inv_D

    def get_hyperparameters(self):

        params = dict()

        params['_init_time'] = self._init_time
        params['_last_time'] = self._last_time

        # The prior function parameters
        for name, param in self.named_parameters():
            # if param.requires_grad:
            params[name.replace(self.__class__.__name__+'__', '')] = param

        params['_nodes_num'] = self._nodes_num
        params['_dim'] = self._dim
        params['_seed'] = self._seed

        return params
