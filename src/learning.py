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
            beta=torch.nn.Parameter(2 * torch.rand(size=(nodes_num,)) - 1, requires_grad=False),
            bins_rwidth=torch.nn.Parameter(torch.zeros(size=(bins_num,)) / float(bins_num), requires_grad=False),
            last_time=last_time,
            seed=seed
        )

        self.__data_loader = data_loader

        self.__lr = learning_rate
        self.__prior_weight = prior_weight
        self.__epochs_num = epochs_num
        self.__verbose = verbose

        # Set the prior function
        self.__neg_log_prior = self.__set_prior(name="gp_kron", kernels=["rbf", "periodic"])

        # Set the correction function
        self.__correction_func = partial(
            self.__correction, centering=True, rotation=True
        )

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__writer = SummaryWriter("../logs/")

        # Order matters for sequential learning
        self.__param_names = [ "x0", "v", "beta", "reg_params", "all"]

        # Set the gradient of bins_rwidth
        self.__set_gradients(**{f"bins_rwidth_grad": True})

    def learn(self, learning_type="seq"):

        # Learns the parameters sequentially
        if learning_type == "seq":

            self.__sequential_learning()

        else:

            raise NotImplementedError("Non-sequential learning not implemented!")

        self.__writer.close()

    def __sequential_learning(self):

        epoch_num_per_var = int(self.__epochs_num / len(self.__param_names))

        for param_idx, param_name in enumerate(self.__param_names):

            # Set the gradients
            self.__set_gradients(**{f"{param_name}_grad": True})

            for epoch in range(param_idx * epoch_num_per_var,
                               min(self.__epochs_num, (param_idx + 1) * epoch_num_per_var)):
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

        # print(node_pairs)
        # nodes = torch.as_tensor([node for nodes in node_pairs for node in nodes])

        # nodes = torch.unique(nodes)
        # if self._v.requires_grad and self.__neg_log_prior is not None:
        #     nll += self.__prior_weight * self.__neg_log_prior(nodes=nodes)

        # Prior term
        nodes = torch.arange(self._nodes_num)
        # if self._v.requires_grad and self.__neg_log_prior is not None:
        if self.__neg_log_prior is not None:
            nll += self.__neg_log_prior(nodes=nodes)

        # if self._v.requires_grad and self.__neg_log_prior is not None:
        #     for j in range(node_pairs.shape[1]):
        #         nll += self.__prior_weight * self.__neg_log_prior(nodes=torch.unique(node_pairs[:, j]))

        # if self._v.requires_grad and self.__neg_log_prior is not None:
        #     nll += self.__prior_weight * self.__neg_log_prior(nodes=torch.unique(node_pairs))

        return nll

    def __set_prior(self, name="", **kwargs):

        if name.lower() == "gp_kron":

            # Parameter initialization
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

        if v_grad is not None:
            self._v.requires_grad = v_grad

        if bins_rwidth_grad is not None:
            self._bins_rwidth.requires_grad = bins_rwidth_grad

        if reg_params_grad is not None:

            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False

            # Set the gradients of the prior function
            for name, param in self.named_parameters():
                if '__prior' in name:
                    param.requires_grad = True

        if all_grad is not None:

            self._beta.requires_grad = all_grad
            self._x0.requires_grad = all_grad
            self._v.requires_grad = all_grad
            self._bins_rwidth.requires_grad = all_grad

            # Set the gradients of the prior function
            for name, param in self.named_parameters():
                if '__prior' in name:
                    param.requires_grad = all_grad

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

    def __get_A(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor,
                get_inv: bool = True, cholesky: bool=True):

        # Compute the inverse of kernel/covariance matrix
        time_mat = bin_centers1 - bin_centers2.transpose(1, 2)
        time_mat = time_mat.squeeze(0)

        kernel = 0.
        for kernel_name in self.__prior_A_kernel_names:
            kernel_func = getattr(self, 'get_'+kernel_name+'_kernel')
            kernel += kernel_func(time_mat=time_mat)

        # If the inverse of the kernel is not required, return only the kernel matrix
        if get_inv is False:
            return kernel

        # Compute the inverse
        if cholesky:
            inv_kernel = torch.cholesky_inverse(torch.linalg.cholesky(kernel))
        else:
            inv_kernel = torch.linalg.inv(kernel)

        return kernel, inv_kernel

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

    def prediction(self, event_times, event_node_pairs, test_middle_point, cholesky=True):

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

        # B: D x D matrix -> self._prior_B_U: D x D upper triangular matrix
        # B = self._prior_B_U @ self._prior_B_U.t()
        B_L = torch.tril(self.__prior_B_L, diagonal=-1) + torch.diag(torch.diag(self.__prior_B_L) ** 2)
        inv_B = torch.cholesky_inverse(B_L)

        # C: N x N matrix
        C_Q = self.__prior_C_Q
        # C_D = torch.diag(self._prior_C_lambda)
        C_inv_D = torch.eye(n=self._nodes_num) #torch.diag(1.0 / self.__prior_C_lambda.expand(len(nodes)))
        # C = C_D + (C_Q @ C_Q.t())
        # By Woodbury matrix identity
        invDQ = C_inv_D @ C_Q
        inv_C = C_inv_D - invDQ @ torch.inverse(torch.eye(C_Q.shape[1]) + C_Q.t() @ C_inv_D @ C_Q) @ invDQ.t()

        # # Compute the inverse of kernel/covariance matrix
        kernel_train_inv = torch.kron(inv_A.contiguous(), torch.kron(inv_B.contiguous(), inv_C.contiguous()))

        # Compute the inverse of kernel/covariance matrix
        A_test_train = self.__get_A(
            bin_centers1=middle_bounds, bin_centers2=test_middle_point.view(1, 1, 1), get_inv=False
        )
        kernel_test_train = torch.kron(A_test_train.contiguous(), torch.kron(torch.linalg.inv(inv_B).contiguous(),
                                                                             torch.linalg.inv(inv_C).contiguous()))

        kernel_test = 1.0 * torch.kron(torch.linalg.inv(inv_B).contiguous(), torch.linalg.inv(inv_C).contiguous())

        batch_v = self._v
        v_vect = self.__vect(batch_v).flatten()

        mean_test = kernel_test_train @ kernel_train_inv @ v_vect
        #sigma_test = kernel_test - kernel_test_train @ kernel_train_inv @ kernel_test_train.t()

        # Second part
        last_v = mean_test.reshape((batch_v.shape[2], batch_v.shape[1])).transpose(-2, -1)

        last_x = self.get_xt(times_list=torch.as_tensor([self._last_time])).squeeze(0)

        delta_last_x = last_x[event_node_pairs[0], :] - last_x[event_node_pairs[1], :]
        delta_v = last_v[event_node_pairs[0], :] - last_v[event_node_pairs[1], :]

        delta_v = delta_v.unsqueeze(0).unsqueeze(0)
        # print(delta_last_x.shape, delta_v.shape, event_times.shape)
        # print(last_x.shape)
        # print(last_x.shape[0], last_x.shape[1])
        delta_xt = delta_last_x + delta_v * event_times.unsqueeze(1)

        norm = torch.norm(delta_xt.squeeze(0).squeeze(0), p=2, dim=1, keepdim=False) ** 2
        non_integral_term = self._beta[event_node_pairs[0]] + self._beta[event_node_pairs[1]] - norm

        integral_term = -self.get_intensity_integral(
            x0=last_x, v=unvectorize(mean_test, size=(1, self._nodes_num, 2)).contiguous(), node_pairs=event_node_pairs,
            bin_bounds=torch.as_tensor([self._last_time, self._last_time+test_middle_point])
        )
        # print(non_integral_term.shape, integral_term.shape)

        return -(non_integral_term + integral_term)

    # def __neg_log_kron_gp_prior(self, nodes, cholesky=True):
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
    #     # # A: T x T matrix
    #     sigma_square = self.__prior_A_sigma * self.__prior_A_sigma
    #     A, inv_A = self.__get_inv_rbf_kernel(
    #         sigma_square, bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=cholesky
    #     )
    #
    #     # # B: D x D matrix -> self._prior_B_U: D x D upper triangular matrix
    #     # B = self._prior_B_U @ self._prior_B_U.t()
    #     B_L = torch.tril(self.__prior_B_L, diagonal=-1) + torch.diag(torch.diag(self.__prior_B_L)**2)
    #     inv_B = torch.cholesky_inverse(B_L)
    #     # # C: N x N matrix
    #     C_Q = self.__prior_C_Q[nodes, :]
    #     # C_D = torch.diag(self._prior_C_lambda)
    #     C_inv_D = torch.diag(1.0 / self.__prior_C_lambda.expand(len(nodes)))
    #     # C = C_D + (C_Q @ C_Q.t())
    #     # By Woodbury matrix identity
    #     invDQ = C_inv_D @ C_Q
    #     inv_C = C_inv_D - invDQ @ torch.inverse(torch.eye(C_Q.shape[1]) + C_Q.t() @ C_inv_D @ C_Q) @ invDQ.t()
    #
    #     # Compute the product in an efficient way
    #     # v^t ( A kron B kron C ) v
    #     batch_v = self._v[:, nodes, :]
    #     v_vect = self.__vect(batch_v).flatten()
    #     p = torch.matmul(
    #         v_vect,
    #         self.__vect(torch.matmul(
    #             self.__vect(torch.matmul(torch.matmul(inv_C.unsqueeze(0), batch_v),
    #                                      inv_B.t().unsqueeze(0))).transpose(0, 1),
    #             inv_A.t()
    #         ))
    #     )
    #
    #     # Compute the log-determinant of the product
    #     final_dim = len(nodes) * self.get_num_of_bins() * self._dim
    #     log_det_kernel = (final_dim / self.get_num_of_bins()) * torch.logdet(A) \
    #                      - (final_dim / self._dim) * torch.logdet(inv_B) \
    #                      - (final_dim / len(nodes)) * torch.logdet(inv_C)
    #
    #     log_prior_likelihood = -0.5 * ( final_dim * math.log(2 * math.pi) + log_det_kernel + p )
    #
    #     return -log_prior_likelihood.squeeze(0)

    # def __get_gp_rbf_kernel(self, sigma_square, cholesky=True):
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
    #     # Compute the inverse of kernel/covariance matrix
    #     time_mat = ((middle_bounds - middle_bounds.transpose(1, 2)) ** 2)
    #
    #     if len(sigma_square) > 1:
    #         time_mat = time_mat.expand(self._dim, bin_num, bin_num)
    #     else:
    #         time_mat = time_mat.squeeze(0)
    #
    #     kernel = torch.exp(-0.5 * torch.div(time_mat, sigma_square))
    #     if cholesky:
    #         inv_kernel = torch.linalg.cholesky(torch.cholesky_inverse(kernel))
    #     else:
    #         inv_kernel = torch.linalg.inv(kernel)
    #
    #     # return torch.log(torch.det(kernel) + 1e-10), inv_kernel
    #     return torch.logdet(kernel), inv_kernel

    # def __neg_log_gp_regularization(self, node_idx, cholesky=True):
    #
    #     # Compute the inverse of kernel/covariance matrix
    #     sigma_square = torch.mul(self.__reg_sigma, self.__reg_sigma).unsqueeze(0).view(self._dim, 1, 1)
    #     log_det_kernel, inv_kernel = self.__get_gp_rbf_kernel(sigma_square, cholesky=cholesky)
    #
    #     # Multiply it by a scaling coefficients
    #     lambda_square = torch.mul(self.__reg_c, self.__reg_c).view(self._dim, 1, 1)
    #     inv_kernel = torch.mul(lambda_square, inv_kernel)
    #
    #     # v: B x N x D tensor
    #     chosen_v = self._v.transpose(0, 2)[:, node_idx, :]
    #     # chosen_v: D x chosen_N x B
    #     # inv_kernel: D x B x B
    #     # log_exp_term: D x chosen_N -> scalar
    #     log_exp_term = -torch.bmm(
    #         chosen_v, inv_kernel
    #     ).mul(chosen_v).sum(dim=2, keepdim=False).sum()
    #     # log_det: D vector
    #     log_det = -0.5 * self.get_num_of_bins() + log_det_kernel.sum() * len(node_idx)
    #     log_pi = -0.5 * torch.log(2 * torch.as_tensor(math.pi)) * len(node_idx) * self._dim
    #
    #     return -(log_pi + log_det + log_exp_term)

    # def __neg_log_kronecker_gp_regularization(self, node_idx, cholesky=True):
    #
    #     # A and inv_A: T x T matrix
    #     sigma_square = self.__reg_sigma * self.__reg_sigma
    #     log_det_A, inv_A = self.__get_gp_rbf_kernel(sigma_square, cholesky=cholesky)
    #
    #     # inv_B: D x D matrix -> inv_BL: DxD lower triangular matrix
    #     inv_B_L = torch.inverse(self.__reg_BL)
    #     inv_B = torch.matmul(inv_B_L.transpose(0, 1), inv_B_L)
    #
    #     # inv_C: N x N matrix
    #     # d = self.__reg_Cd[node_idx]
    #     d = torch.ones(size=(len(node_idx), ), dtype=torch.float)
    #
    #     V = torch.mm(torch.diag(torch.div(1.0, d)), torch.mm(self.__reg_CU[node_idx, :], self.__reg_CR))  #self.__reg_CV[node_idx, :]
    #     inv_C = torch.diag(torch.div(1.0, d)) - torch.mm(V, V.transpose(0, 1))
    #
    #     log_det_kernel = + (self._dim * len(node_idx)) * log_det_A \
    #                      - (self.get_num_of_bins()*len(node_idx)) * torch.logdet(inv_B) \
    #                      - (self.get_num_of_bins()*self._dim) * torch.logdet(inv_C)
    #
    #     inv_kernel = torch.kron(inv_A.contiguous(), torch.kron(inv_B, inv_C))
    #
    #     # v: B x N x D tensor
    #     # chosen: B x chosen_N x D tensor ->
    #     chosen = self._v.transpose(1, 2)[:, :, node_idx].flatten()
    #
    #     result = -0.5 * self._dim * len(node_idx) * self.get_num_of_bins() * ( log_det_kernel + 2 * math.pi) \
    #              - torch.mul(torch.mul(chosen, inv_kernel), chosen)
    #
    #     return -result.sum()

    # def __get_inv_rbf_kernel(self, sigma_square, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor, inv=True, cholesky=True):
    #
    #     # Compute the inverse of kernel/covariance matrix
    #     time_mat = ((bin_centers1 - bin_centers2.transpose(1, 2)) ** 2)
    #
    #     if len(sigma_square) > 1:
    #         time_mat = time_mat.expand(self._dim, len(bin_centers1), len(bin_centers2))
    #     else:
    #         time_mat = time_mat.squeeze(0)
    #
    #     kernel = torch.exp(-0.5 * torch.div(time_mat, sigma_square))
    #
    #     if inv is False:
    #         return kernel
    #
    #     if cholesky:
    #         inv_kernel = torch.cholesky_inverse(torch.linalg.cholesky(kernel))
    #     else:
    #         inv_kernel = torch.linalg.inv(kernel)
    #
    #     # return torch.log(torch.det(kernel) + 1e-10), inv_kernel
    #     return kernel, inv_kernel

    # def __set_neg_log_prior(self, name: str = None):
    #
    #     if name.lower() == "gp_kron":
    #
    #         # Parameter initialization
    #         self.__prior_k = 2
    #         self.__prior_A_sigma = torch.nn.Parameter(
    #             2 * torch.rand(size=(1,)) - 1, requires_grad=False
    #         )
    #         # self._prior_B_U is upper triangular matrix with positive diagonal entries
    #         self.__prior_B_L = torch.nn.Parameter(
    #             torch.tril(2 * torch.rand(size=(self._dim, self._dim)) - 1, diagonal=-1) +
    #             torch.diag(torch.rand(size=(self._dim, ))),
    #             requires_grad=False
    #         )
    #
    #         self.__prior_C_Q = torch.nn.Parameter(
    #             2 * torch.rand(size=(self._nodes_num, self.__prior_k)) - 1, requires_grad=False
    #         )
    #         self.__prior_C_lambda = torch.nn.Parameter(
    #             2 * torch.rand(size=(1, )), requires_grad=False
    #         )
    #
    #         # Return the prior function
    #         return partial(self.__neg_log_kron_gp_prior, cholesky=True)
    #
    #     else:
    #
    #         raise ValueError("Invalid prior name!")