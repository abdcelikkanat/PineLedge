import sys
import math
import torch
from src.base import BaseModel
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from utils import mean_normalization
import utils
import time

COUNTER = 0


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data_loader, nodes_num, bins_num, dim, last_time: float,
                 learning_rate: float, prior_weight: float = 1.0, epochs_num: int = 100, steps_per_epoch=10,
                 device: torch.device = "cpu", verbose: bool = False, seed: int = 0):

        super(LearningModel, self).__init__(
            x0=torch.nn.Parameter(2 * torch.rand(size=(nodes_num, dim), device=device) - 1, requires_grad=False),
            v=torch.nn.Parameter(2 * torch.rand(size=(bins_num, nodes_num, dim), device=device) - 1, requires_grad=False),
            beta=torch.nn.Parameter(2 * torch.zeros(size=(nodes_num,), device=device), requires_grad=False),
            bins_rwidth=torch.nn.Parameter(torch.zeros(size=(bins_num,), device=device) / float(bins_num), requires_grad=False),
            last_time=last_time,
            device=device,
            verbose=verbose,
            seed=seed
        )

        self.initialize_prior_params()

        self.__data_loader = data_loader

        # Set the prior function
        # self.__neg_log_prior = self.__set_prior( kernels=["rbf",]) # rbf periodic

        # Set the correction function
        self.__correction_func = None

        self.__learning_procedure = "alt"  #"seq"
        self.__learning_rate = learning_rate
        self.__epochs_num = epochs_num
        self.__steps_per_epoch = steps_per_epoch
        self.__prior_weight = prior_weight

        #self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__learning_rate)
        self.__optimizer = None

        self.__verbose = verbose
        self.__device = device
        self.__writer = SummaryWriter("../experiments/logs/determinant")

        # Order matters for sequential learning
        self.__learning_param_names = [ ["x0", "v"], ["reg_params"] ]  # , ["reg_params"] ["bins_rwidth"] "reg_params" "bins_rwidth" ["v", "bins_rwidth"], ["beta"], ["bins_rwidth"]
        self.__learning_param_epoch_weights = [2, 1]  # 2

        self.__add_prior = False  # Do not change

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

            self.__alternating_learning()

        else:

            raise NotImplementedError("Non-sequential learning not implemented!")

        if self.__writer is not None:
            self.__writer.close()

    def __sequential_learning(self):

        # epoch_num_per_var = int(self.__epochs_num / len(self.__param_names))
        epoch_cumsum = torch.cumsum(torch.as_tensor([0] + self.__param_epoch_weights), dim=0)
        epoch_num_per_var = (self.__epochs_num * epoch_cumsum // torch.sum(torch.as_tensor(self.__param_epoch_weights))).type(torch.int)

        for param_idx, param_names in enumerate(self.__param_names):

            # Set the gradients
            if type(param_names) is not list:
                param_names = [param_names]

            for pname in param_names:
                self.__set_gradients(**{f"{pname}_grad": True})

            # for epoch in range(param_idx * epoch_num_per_var,
            #                    min(self.__epochs_num, (param_idx + 1) * epoch_num_per_var)):
            for epoch in range(epoch_num_per_var[param_idx], epoch_num_per_var[param_idx+1]):
                self.__train_one_epoch(
                    epoch=epoch, correction_func=self.__correction_func
                )

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
        # for batch_node_pairs, batch_times_list, in self.__data_loader:
        #
        #     if batch_num == self.__steps_per_epoch:
        #         break
        for _ in range(self.__steps_per_epoch):

            self.train()

            batch_node_pairs, batch_times_list = next(iter(self.__data_loader))
            batch_node_pairs, batch_times_list = batch_node_pairs.to(self._device), batch_times_list

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

        # if self.__writer is not None:
        #     self.__writer.add_scalar(tag="Loss/train", scalar_value=average_epoch_loss, global_step=epoch)

        if self.__verbose and (epoch % 10 == 0 or epoch == self.__epochs_num - 1):
            print(f"| Epoch = {epoch} | Loss/train: {average_epoch_loss} | Elapsed time: {time.time() - init_time}")

    def forward(self, time_seq_list, node_pairs):

        nll = self.get_negative_log_likelihood(time_seq_list, node_pairs)

        nll += self.__prior_weight * self.__neg_log_prior(nodes=torch.unique(node_pairs))

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # print("x0: ", self._x0.grad, self._x0[0, :])
        # print(self.__prior_weight, self.get_negative_log_likelihood(time_seq_list, node_pairs), self.__neg_log_prior(nodes=torch.unique(node_pairs)))

        return nll

    def initialize_prior_params(self, scale_const: float = 1e0, kernel_names: list = None):

        # Initialize the prior terms
        # Set the parameters required for the construction of the matrix A
        self.__kernel_names = ["rbf"] if kernel_names is None else kernel_names

        for kernel_name in self.__kernel_names:

            if kernel_name == "rbf":
                # Output variance
                if len(self.__kernel_names) > 1:
                    self.__prior_rbf_sigma = torch.nn.Parameter(
                        2 * scale_const * torch.rand(size=(1,)) - scale_const, requires_grad=False
                    )
                # Length scale
                self.__prior_rbf_l = torch.nn.Parameter(
                    2 * torch.rand(size=(1,)) - 1, requires_grad=False
                )

            elif kernel_name == "periodic":
                # Output variance
                if len(self.__kernel_names) > 1:
                    self.__prior_periodic_sigma = torch.nn.Parameter(
                        2 * scale_const * torch.rand(size=(1,)) - scale_const, requires_grad=False
                    )
                # Period
                self.__prior_periodic_p = torch.nn.Parameter(
                    2 * torch.rand(size=(1,)) - 1, requires_grad=False
                )
                # Length scale
                self.__prior_periodic_l = torch.nn.Parameter(
                    2 * torch.rand(size=(1,)) - 1, requires_grad=False
                )

            else:

                raise ValueError("Invalid kernel name")

        # Set the noise term for the kernel
        self.__prior_kernel_noise = torch.nn.Parameter(2 * scale_const * torch.rand(size=(1,)) - 1, requires_grad=False)

        # Set the parameters required for the construction of the matrix B
        # self._prior_B_L is lower triangular matrix with positive diagonal entries
        self.__prior_B_L = torch.nn.Parameter(
            torch.tril(2 * scale_const * torch.rand(size=(self._dim, self._dim)) - scale_const, diagonal=-1) +
            scale_const * torch.diag(torch.rand(size=(self._dim,))),
            requires_grad=False
        )
        # Set the noise term for the kernel
        self.__prior_B_noise = torch.nn.Parameter(
            (2.0 / scale_const) * torch.rand(size=(1,)) - (1.0 / scale_const), requires_grad=False
        )

        # Set the parameters required for the construction of the matrix C
        self.__prior_C_Q_dim = 2
        self.__prior_C_Q = torch.nn.Parameter(
            (2.0 / scale_const) * torch.rand(size=(self._nodes_num, self.__prior_C_Q_dim)) - (1.0 / scale_const), requires_grad=False
        )
        self.__prior_C_lambda = torch.nn.Parameter(
            (2.0 / scale_const) * torch.rand(size=(1,)) - (1.0 / scale_const), requires_grad=False
        )

    # def __set_prior(self, kernels: list = None):
    #
    #
    #
    #     # Return the prior function
    #     return partial(self.__neg_log_prior, cholesky=True)

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None, bins_rwidth_grad=None, reg_params_grad=None):

        if beta_grad is not None:
            self._beta.requires_grad = beta_grad

        if x0_grad is not None:
            self._x0.requires_grad = x0_grad

        if v_grad is not None:
            self._v.requires_grad = v_grad

        if bins_rwidth_grad is not None:
            self._bins_rwidth.requires_grad = bins_rwidth_grad

        if reg_params_grad is not None:

            # Set the gradients of the prior function
            for name, param in self.named_parameters():
                if '__prior' in name:
                    param.requires_grad = reg_params_grad

    # def __correction(self, centering=None, rotation=None):
    #
    #     with torch.no_grad():
    #
    #         if centering:
    #             x0_m = torch.mean(self._x0, dim=0, keepdim=True)
    #             self._x0 -= x0_m
    #
    #             v_m = torch.mean(self._v, dim=1, keepdim=True)
    #             self._v -= v_m
    #
    #             # v_m = self._v.sum(dim=0, keepdim=True).sum(dim=1, keepdim=True) / (self._v.shape[0]*self._v.shape[1])
    #             # self._v -= v_m
    #
    #         if rotation:
    #             U, S, _ = torch.linalg.svd(
    #                 torch.vstack((self._x0, self._v.view(-1, self._v.shape[2]))),
    #                 full_matrices=False
    #             )
    #
    #             temp = torch.mm(U, torch.diag(S))
    #
    #             self._x0.data = temp[:self._x0.shape[0], :]
    #             self._v.data = temp[self._x0.shape[0]:, :].view(self._v.shape[0], self._v.shape[1], self._v.shape[2])

    def get_rbf_kernel(self, time_mat):

        kernel = torch.exp(-0.5 * torch.div(time_mat**2, self.__prior_rbf_l**2))

        if len(self.__kernel_names) > 1:
            kernel = (self.__prior_kernel_sigma ** 2) * kernel

        return kernel

    def get_periodic_kernel(self, time_mat):

        kernel = torch.exp(
            -2 * torch.sin(math.pi * torch.abs(time_mat) / self.__prior_periodic_p)**2 / (self.__prior_periodic_l**2)
        )
        if len(self.__kernel_names) > 1:
            kernel = (self.__prior_periodic_sigma**2) * kernel
            
        return kernel

    def __get_A(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor,
                get_inv: bool = True, cholesky: bool = True):

        # Compute the inverse of kernel/covariance matrix
        time_mat = bin_centers1 - bin_centers2.transpose(1, 2)
        time_mat = time_mat.squeeze(0)

        kernel = 0.
        for kernel_name in self.__kernel_names:
            kernel_func = getattr(self, 'get_'+kernel_name+'_kernel')
            kernel += kernel_func(time_mat=time_mat)

        # Add a noise term
        kernel += torch.eye(n=kernel.shape[0], m=kernel.shape[1]) * (self.__prior_kernel_noise**2)

        # If the inverse of the kernel is not required, return only the kernel matrix
        if not get_inv:
            return kernel

        # Compute the inverse
        if cholesky:
            inv_kernel = torch.cholesky_inverse(torch.linalg.cholesky(kernel))
        else:
            inv_kernel = torch.linalg.inv(kernel)

        return kernel, inv_kernel

    def __get_B(self):

        # B: D x D matrix -> self._prior_B_L: D x D lower triangular matrix
        # B = self._prior_B_L @ self._prior_B_L.t()
        B_L = torch.tril(self.__prior_B_L, diagonal=-1) + torch.diag(torch.diag(self.__prior_B_L)**2)

        # Definition of inverse
        inv_B = torch.cholesky_inverse(B_L) + torch.eye(n=self._dim, m=self._dim) * (self.__prior_B_noise**2)
        # Add a noise
        # inv_B += torch.eye(n=inv_B.shape[0], m=inv_B.shape[1]) * (self.__prior_B_noise**2)

        return inv_B

    def __get_C(self, nodes = None):

        # C: N x N matrix
        if nodes is None:
            C_Q = self.__prior_C_Q
            nodes_num = C_Q.shape[0]
        else:
            C_Q = self.__prior_C_Q[nodes, :]
            nodes_num = len(nodes)

        C_D = torch.diag(self.__prior_C_lambda.expand(nodes_num) ** 2)

        inv_C = C_D + C_Q @ C_Q.t()
        return inv_C

    def __neg_log_prior(self, nodes, cholesky=True):

        # Get the number of bin size
        bin_num = self.get_num_of_bins()

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, bin_num) / 2.

        # A: T x T matrix
        A, inv_A = self.__get_A(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=cholesky
        )

        # B: D x D matrix
        inv_B = self.__get_B()

        # C: len(nodes) x len(nodes) matrix
        inv_C = self.__get_C(nodes=nodes)

        # Compute the product, v^t ( A kron B kron C )^-1 v,  in an efficient way
        batch_v = mean_normalization(self._v)[:, nodes, :]
        v_vect = utils.vectorize(batch_v).flatten()
        p = v_vect @ utils.vectorize(
            utils.vectorize((inv_C.unsqueeze(0) @ batch_v) @ inv_B.t().unsqueeze(0)).t() @ inv_A.t()
        )

        # Compute the log-determinant of the product
        final_dim = len(nodes) * self.get_num_of_bins() * self._dim
        log_det_kernel = (final_dim / self.get_num_of_bins()) * torch.logdet(A) \
                         - (final_dim / self._dim) * torch.logdet(inv_B) \
                         - (final_dim / len(nodes)) * torch.logdet(inv_C)

        global COUNTER
        self.__writer.add_scalar(tag="Determinant", scalar_value=log_det_kernel.item(), global_step=COUNTER)
        COUNTER += 1

        log_prior_likelihood = -0.5 * (final_dim * math.log(2 * math.pi) + log_det_kernel + p)
        #log_prior_likelihood = torch.logdet(A) + torch.logdet(inv_B)
        return -log_prior_likelihood.squeeze(0)

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

    # def get_beta_kernel(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor,
    #             get_inv: bool = True, cholesky: bool=True):
    #
    #     # Compute the inverse of kernel/covariance matrix
    #     time_mat = bin_centers1 - bin_centers2.transpose(0, 1)
    #     # time_mat = time_mat.squeeze(0)
    #
    #     kernel = self.get_rbf_kernel(time_mat=time_mat)
    #
    #     # Add a noise term
    #     kernel += torch.eye(n=kernel.shape[0], m=kernel.shape[1]) * (self.__prior_kernel_noise_sigma**2)
    #
    #     # If the inverse of the kernel is not required, return only the kernel matrix
    #     if get_inv is False:
    #         return kernel
    #
    #     # Compute the inverse
    #     if cholesky:
    #         # print("=", self.__prior_rbf_l**2, self.__prior_rbf_sigma**2)
    #         # print(kernel)
    #         inv_kernel = torch.cholesky_inverse(torch.linalg.cholesky(kernel))
    #     else:
    #         inv_kernel = torch.linalg.inv(kernel)
    #
    #     return kernel, inv_kernel
    #
    # def beta_prior(self, nodes):
    #
    #     # Get the number of bin size
    #     bin_num = self.get_num_of_bins()
    #
    #     # Get the bin bounds
    #     bounds = self.get_bins_bounds()
    #
    #     # Get the middle time points of the bins for TxT covariance matrix
    #     middle_bounds = (bounds[1:] + bounds[:-1]).view(1, bin_num) / 2.
    #
    #     # K: T x T matrix
    #     K, inv_K = self.get_beta_kernel(
    #         bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=True
    #     )
    #
    #     p = torch.matmul(self._beta[:, nodes].transpose(0, 1), torch.matmul(inv_K, self._beta[:, nodes])).sum()
    #     # Compute the log-determinant of the product
    #     final_dim = K.shape[0]
    #     log_det_kernel = torch.logdet(K)
    #
    #     log_prior_likelihood = -0.5 * (final_dim * math.log(2 * math.pi) + log_det_kernel + p)
    #
    #     return -log_prior_likelihood.squeeze(0)

    # def __set_prior(self, name="", **kwargs):
    #
    #     if name.lower() == "gp_kron":
    #
    #         # Parameter initialization
    #         self.__prior_kernel_noise_sigma = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)
    #
    #         # Set the parameters required for the construction of the matrix A
    #         self.__prior_A_kernel_names = kwargs["kernels"]
    #         for kernel_name in self.__prior_A_kernel_names:
    #
    #             if kernel_name == "rbf":
    #
    #                 self.__prior_rbf_sigma = torch.nn.Parameter(200 * torch.rand(size=(1,)) - 100, requires_grad=False)
    #                 self.__prior_rbf_l = 2*torch.nn.Parameter(2 * torch.rand(size=(1,)) , requires_grad=False)
    #
    #             elif kernel_name == "periodic":
    #
    #                 self.__prior_periodic_sigma = torch.nn.Parameter(200 * torch.rand(size=(1,)) - 100, requires_grad=False)
    #                 self.__prior_periodic_p = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)
    #                 self.__prior_periodic_l = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)
    #
    #             else:
    #
    #                 raise ValueError("Invalid kernel name")
    #
    #         # Set the parameters required for the construction of the matrix B
    #         # self._prior_B_L is lower triangular matrix with positive diagonal entries
    #         self.__prior_B_L = torch.nn.Parameter(
    #             torch.tril(200 * torch.rand(size=(self._dim, self._dim)) - 100, diagonal=-1) +
    #             100*torch.diag(torch.rand(size=(self._dim, ))),
    #             requires_grad=False
    #         )
    #
    #         # Set the parameters required for the construction of the matrix C
    #         self.__prior_C_Q_dim = 2
    #         self.__prior_C_Q = torch.nn.Parameter(
    #             2 * torch.rand(size=(self._nodes_num, self.__prior_C_Q_dim)) - 1, requires_grad=False
    #         )
    #         self.__prior_C_lambda = torch.nn.Parameter(
    #             2 * torch.rand(size=(1, )) - 1, requires_grad=False
    #         )
    #
    #         # Return the prior function
    #         return partial(self.__neg_log_prior, cholesky=True)
    #
    #     else:
    #
    #         raise ValueError("Invalid prior name!")
