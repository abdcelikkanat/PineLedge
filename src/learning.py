import sys
import math
import torch
from src.base import BaseModel
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from utils.utils import unvectorize
import time


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data_loader, nodes_num, bins_num, dim, last_time: float,
                 learning_rate: float, prior_weight: float = 1.0, epochs_num: int = 100, verbose: bool = False,
                 seed: int = 0):

        super(LearningModel, self).__init__(
            x0=torch.nn.Parameter(2 * torch.rand(size=(nodes_num, dim)) - 1, requires_grad=False),
            v=torch.nn.Parameter(2 * torch.rand(size=(bins_num, nodes_num, dim)) - 1, requires_grad=False),
            beta=torch.nn.Parameter(2 * torch.zeros(size=(nodes_num, )) - 1, requires_grad=False),
            bins_rwidth=torch.nn.Parameter(torch.zeros(size=(bins_num,)) / float(bins_num), requires_grad=False),
            alpha=torch.nn.Parameter(2 * torch.rand(size=(nodes_num*(nodes_num-1)//2, bins_num)) - 1, requires_grad=False),
            last_time=last_time,
            seed=seed
        )

        self.__data_loader = data_loader

        self.__lr = learning_rate
        # self.__prior_weight = prior_weight
        self.__epochs_num = epochs_num
        self.__verbose = verbose

        # Set the prior function
        self.__neg_log_prior = self.__set_prior(name="gp_kron", kernels=["rbf",]) # rbf periodic

        # Set the correction function
        self.__correction_func = partial(
            self.__correction, centering=True, rotation=True
        )

        self.__learning_type = "alt" #"seq"
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__writer = SummaryWriter("../logs/")

        # Order matters for sequential learning
        self.__param_names = [ "x0",  ["v",  "bins_rwidth",], "beta",]  # "reg_params"
        self.__param_epoch_weights = [1, 1, 1]  # 2

        # Set the gradient of bins_rwidth
        # self.__set_gradients(**{f"bins_rwidth_grad": True})

        self.__add_prior = False  # Do not change

    def learn(self, learning_type=None):

        learning_type = self.__learning_type if learning_type is None else learning_type

        # Learns the parameters sequentially
        if learning_type == "seq":

            self.__sequential_learning()

        elif learning_type == "alt":

            # print(list(self.parameters()))
            self.__alternating_learning()
        else:

            raise NotImplementedError("Non-sequential learning not implemented!")

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

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)

        param_names_list = [[param_names] if type(param_names) is not list else param_names for param_names in self.__param_names]

        for epoch in range(self.__epochs_num):
            # Make all false
            for pname_sublist in param_names_list:
                for pname in pname_sublist:
                    self.__set_gradients(**{f"{pname}_grad": False})

            remainder = epoch % len(self.__param_names)
            pname_sublist = param_names_list[remainder]
            for pname in pname_sublist:
                self.__set_gradients(**{f"{pname}_grad": True})

            self.__train_one_epoch(
                epoch=epoch, correction_func=self.__correction_func
            )

    def __train_one_epoch(self, epoch, correction_func=None):

        self.train()

        init_time = time.time()

        average_epoch_loss = 0
        for batch_node_pairs, batch_times_list, in self.__data_loader:
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
            self.__optimizer.zero_grad()

            # Backward pass
            batch_loss_sum.backward()

            # Perform a step
            self.__optimizer.step()

            if correction_func is not None:
                correction_func()

        average_epoch_loss = average_epoch_loss / len(self.__data_loader)

        self.__writer.add_scalar(tag="Loss/train", scalar_value=average_epoch_loss, global_step=epoch)

        if self.__verbose and (epoch % 10 == 0 or epoch == self.__epochs_num - 1):
            print(f"| Epoch = {epoch} | Loss/train: {average_epoch_loss} | {time.time() - init_time}")

    def forward(self, time_seq_list, node_pairs):

        nll = self.get_negative_log_likelihood(time_seq_list, node_pairs)

        # Prior term
        nodes = torch.unique(node_pairs)
        # if self._v.requires_grad and self.__neg_log_prior is not None:

        if self.__add_prior:
            nll += self.__neg_log_prior(nodes=nodes)
            # print(nll.shape, self.beta_prior(nodes=nodes).shape)
            # nll += self.beta_prior(nodes=nodes)

        return nll

    def __set_prior(self, name="", **kwargs):

        if name.lower() == "gp_kron":

            # Parameter initialization
            self.__prior_kernel_noise_sigma = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)

            # Set the parameters required for the construction of the matrix A
            self.__prior_A_kernel_names = kwargs["kernels"]
            for kernel_name in self.__prior_A_kernel_names:

                if kernel_name == "rbf":

                    self.__prior_rbf_sigma = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)
                    self.__prior_rbf_l = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)

                elif kernel_name == "periodic":

                    self.__prior_periodic_sigma = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)
                    self.__prior_periodic_p = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)
                    self.__prior_periodic_l = torch.nn.Parameter(2 * torch.rand(size=(1,)) - 1, requires_grad=False)

                else:

                    raise ValueError("Invalid kernel name")

            # Set the parameters required for the construction of the matrix B
            # self._prior_B_U is upper triangular matrix with positive diagonal entries
            self.__prior_B_L = torch.nn.Parameter(
                torch.tril(2 * torch.rand(size=(self._dim, self._dim)) - 1, diagonal=-1) +
                torch.diag(torch.rand(size=(self._dim, ))),
                requires_grad=False
            )

            # Set the parameters required for the construction of the matrix C
            self.__prior_C_Q_dim = 2
            self.__prior_C_Q = torch.nn.Parameter(
                2 * torch.rand(size=(self._nodes_num, self.__prior_C_Q_dim)) - 1, requires_grad=False
            )
            self.__prior_C_lambda = torch.nn.Parameter(
                2 * torch.rand(size=(1, )) - 1, requires_grad=False
            )

            # Return the prior function
            return partial(self.__neg_log_prior, cholesky=True)

        else:

            raise ValueError("Invalid prior name!")

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None, bins_rwidth_grad=None, reg_params_grad=None, all_grad=None):

        if beta_grad is not None:
            self._beta.requires_grad = beta_grad

        if x0_grad is not None:
            self._x0.requires_grad = x0_grad
            # self._alpha.requires_grad = True

        if v_grad is not None:
            self._v.requires_grad = v_grad

        if bins_rwidth_grad is not None:
            self._bins_rwidth.requires_grad = bins_rwidth_grad

        # if reg_params_grad is not None:
        #
        #     self.__add_prior = True
        #
        #     for name, param in self.named_parameters():
        #         if param.requires_grad:
        #             param.requires_grad = False
        #
        #     # Set the gradients of the prior function
        #     for name, param in self.named_parameters():
        #         if '__prior' in name:
        #             param.requires_grad = True

        if reg_params_grad is not None:

            self.__add_prior = reg_params_grad

            # Set the gradients of the prior function
            for name, param in self.named_parameters():
                if '__prior' in name:
                    param.requires_grad = reg_params_grad

        # if all_grad is not None:
        #
        #     self.__add_prior = True
        #
        #     self._beta.requires_grad = all_grad
        #     self._x0.requires_grad = all_grad
        #     self._v.requires_grad = all_grad
        #     self._bins_rwidth.requires_grad = all_grad
        #     # self._alpha.requires_grad = True
        #
        #     # Set the gradients of the prior function
        #     for name, param in self.named_parameters():
        #         if '__prior' in name:
        #             param.requires_grad = all_grad

    def __correction(self, centering=None, rotation=None):

        with torch.no_grad():

            if centering:
                x0_m = torch.mean(self._x0, dim=0, keepdim=True)
                self._x0 -= x0_m

                v_m = torch.mean(self._v, dim=1, keepdim=True)
                self._v -= v_m

            if rotation:
                U, S, _ = torch.linalg.svd(
                    torch.vstack((self._x0, self._v.view(-1, self._v.shape[2]))),
                    full_matrices=False
                )

                temp = torch.mm(U, torch.diag(S))

                self._x0.data = temp[:self._x0.shape[0], :]
                self._v.data = temp[self._x0.shape[0]:, :].view(self._v.shape[0], self._v.shape[1], self._v.shape[2])

    def __vect(self, x):

        return x.transpose(-2, -1).flatten(-2)

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

    def get_rbf_kernel(self, time_mat):

        kernel = self.__prior_rbf_sigma**2 * torch.exp(
            -0.5 * torch.div(time_mat**2, self.__prior_rbf_l**2)
        )

        return kernel

    def get_periodic_kernel(self, time_mat):

        kernel = self.__prior_periodic_sigma**2 * torch.exp(
            -2 * torch.sin(math.pi * torch.abs(time_mat) / self.__prior_periodic_p)**2 / (self.__prior_periodic_l**2)
        )

        return kernel

    def get_beta_kernel(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor,
                get_inv: bool = True, cholesky: bool=True):

        # Compute the inverse of kernel/covariance matrix
        time_mat = bin_centers1 - bin_centers2.transpose(0, 1)
        # time_mat = time_mat.squeeze(0)

        kernel = self.get_rbf_kernel(time_mat=time_mat)

        # Add a noise term
        kernel += torch.eye(n=kernel.shape[0], m=kernel.shape[1]) * (self.__prior_kernel_noise_sigma**2)

        # If the inverse of the kernel is not required, return only the kernel matrix
        if get_inv is False:
            return kernel

        # Compute the inverse
        if cholesky:
            # print("=", self.__prior_rbf_l**2, self.__prior_rbf_sigma**2)
            # print(kernel)
            inv_kernel = torch.cholesky_inverse(torch.linalg.cholesky(kernel))
        else:
            inv_kernel = torch.linalg.inv(kernel)

        return kernel, inv_kernel

    def beta_prior(self, nodes):

        # Get the number of bin size
        bin_num = self.get_num_of_bins()

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (bounds[1:] + bounds[:-1]).view(1, bin_num) / 2.

        # K: T x T matrix
        K, inv_K = self.get_beta_kernel(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=True
        )

        p = torch.matmul(self._beta[:, nodes].transpose(0, 1), torch.matmul(inv_K, self._beta[:, nodes])).sum()
        # Compute the log-determinant of the product
        final_dim = K.shape[0]
        log_det_kernel = torch.logdet(K)

        log_prior_likelihood = -0.5 * (final_dim * math.log(2 * math.pi) + log_det_kernel + p)

        return -log_prior_likelihood.squeeze(0)

    def __get_A(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor,
                get_inv: bool = True, cholesky: bool=True):

        # Compute the inverse of kernel/covariance matrix
        time_mat = bin_centers1 - bin_centers2.transpose(1, 2)
        time_mat = time_mat.squeeze(0)

        kernel = 0.
        for kernel_name in self.__prior_A_kernel_names:
            kernel_func = getattr(self, 'get_'+kernel_name+'_kernel')
            kernel += kernel_func(time_mat=time_mat)

        # Add a noise term
        kernel += torch.eye(n=kernel.shape[0], m=kernel.shape[1]) * (self.__prior_kernel_noise_sigma**2)

        # If the inverse of the kernel is not required, return only the kernel matrix
        if get_inv is False:
            return kernel

        # Compute the inverse
        if cholesky:
            # print("=", self.__prior_rbf_l**2, self.__prior_rbf_sigma**2)
            # print(kernel)
            inv_kernel = torch.cholesky_inverse(torch.linalg.cholesky(kernel))
        else:
            inv_kernel = torch.linalg.inv(kernel)

        return kernel, inv_kernel
        # return 1e+6*torch.eye(n=kernel.shape[0]), 1e-6*torch.eye(n=kernel.shape[0])

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
        # A = torch.eye(n=self.get_num_of_bins())*1
        # inv_A = torch.eye(n=self.get_num_of_bins())*1

        # B: D x D matrix -> self._prior_B_U: D x D upper triangular matrix
        # B = self._prior_B_U @ self._prior_B_U.t()
        B_L = torch.tril(self.__prior_B_L, diagonal=-1) + torch.diag(torch.diag(self.__prior_B_L) ** 2)
        inv_B = torch.cholesky_inverse(B_L)
        # inv_B = torch.eye(n=self._dim)

        # C: N x N matrix
        C_Q = self.__prior_C_Q[nodes, :]
        # C_D = torch.diag(self._prior_C_lambda)
        C_inv_D = torch.eye(n=len(nodes)) #torch.diag(1.0 / self.__prior_C_lambda.expand(len(nodes)))
        # C = C_D + (C_Q @ C_Q.t())
        # By Woodbury matrix identity
        invDQ = C_inv_D @ C_Q
        inv_C = C_inv_D - invDQ @ torch.inverse(torch.eye(C_Q.shape[1]) + C_Q.t() @ C_inv_D @ C_Q) @ invDQ.t()
        # inv_C = torch.eye(n=len(nodes))

        # Compute the product in an efficient way
        # v^t ( A kron B kron C ) v
        batch_v = self._v[:, nodes, :]
        v_vect = self.__vect(batch_v).flatten()
        p = torch.matmul(
            v_vect,
            self.__vect(torch.matmul(
                self.__vect(torch.matmul(torch.matmul(inv_C.unsqueeze(0), batch_v),
                                         inv_B.t().unsqueeze(0))).transpose(0, 1),
                inv_A.t()
            ))
        )

        # Compute the log-determinant of the product
        final_dim = len(nodes) * self.get_num_of_bins() * self._dim
        log_det_kernel = (final_dim / self.get_num_of_bins()) * torch.logdet(A) \
                         - (final_dim / self._dim) * torch.logdet(inv_B) \
                         - (final_dim / len(nodes)) * torch.logdet(inv_C)

        log_prior_likelihood = -0.5 * (final_dim * math.log(2 * math.pi) + log_det_kernel + p)

        return -log_prior_likelihood.squeeze(0)
