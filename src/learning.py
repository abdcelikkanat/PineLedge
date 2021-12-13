import sys
import math
import torch
from src.base import BaseModel
from torch.utils.tensorboard import SummaryWriter


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data_loader, nodes_num, bins_num, dim, last_time: float,
                 learning_rate: float, reg_lambda: float = 0.01, epochs_num: int = 100, verbose: bool = False,
                 seed: int = 0):

        super(LearningModel, self).__init__(
            x0=torch.nn.Parameter(2 * torch.rand(size=(nodes_num, dim)) - 1, requires_grad=False),
            v=torch.nn.Parameter(2 * torch.rand(size=(bins_num, nodes_num, dim)) - 1, requires_grad=False),
            beta=torch.nn.Parameter(2 * torch.rand(size=(nodes_num,)) - 1, requires_grad=False),
            bins_rwidth=torch.nn.Parameter(torch.ones(size=(bins_num,)) / float(bins_num), requires_grad=False),
            last_time=last_time,
            seed=seed
        )

        self.__data_loader = data_loader

        self.__lr = learning_rate
        self.__reg_lambda = reg_lambda
        self.__epochs_num = epochs_num
        self.__verbose = verbose

        # Regularization function
        self.__current_reg_lambda = 0

        self.__regularization_func = None
        self.__reg_kron = True

        if self.__reg_kron:
            self.__reg_k = 2
            self.__reg_sigma = torch.nn.Parameter(2 * torch.rand(size=(1, )) - 1, requires_grad=False)
            self.__reg_BL = torch.nn.Parameter(2 * torch.rand(size=(self._dim, self._dim)) - 1, requires_grad=False)
            self.__reg_Cd = torch.nn.Parameter(2 * torch.rand(size=(self._nodes_num, )) - 1, requires_grad=False)
            self.__reg_CU = torch.nn.Parameter(2 * torch.rand(size=(self._nodes_num, self.__reg_k)) - 1, requires_grad=False)
            self.__reg_CR = torch.nn.Parameter(2 * torch.rand(size=(self.__reg_k, self.__reg_k) ) - 1, requires_grad=False)
            #self.__reg_CV = torch.nn.Parameter(2 * torch.rand(size=(self._nodes_num, self.__reg_k)) - 1, requires_grad=False)
            self.__regularization_func = self.__neg_log_kronecker_gp_regularization

        else:
            self.__reg_sigma = torch.nn.Parameter(2 * torch.rand(size=(dim, )) - 1, requires_grad=False)
            self.__reg_c = torch.nn.Parameter(2 * torch.rand(size=(dim,)) - 1, requires_grad=False)
            self.__regularization_func = self.__neg_log_gp_regularization

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__writer = SummaryWriter("../logs/")

        # Order matters for sequential learning
        self.__param_names = ["x0", "v", "beta"]

    def learn(self, learning_type="seq"):

        # Learns the parameters sequentially
        if learning_type == "seq":

            self.__sequential_learning()

        else:

            raise NotImplementedError("Non-sequential learning not implemented!")

        self.__writer.close()

    def __sequential_learning(self):

        # Set the gradient of bins_rwidth
        self.__set_gradients(**{f"bins_rwidth_grad": True})

        epoch_num_per_var = int(self.__epochs_num / len(self.__param_names))

        for param_idx, param_name in enumerate(self.__param_names):

            # Set the gradients
            self.__set_gradients(**{f"{param_name}_grad": True})

            for epoch in range(param_idx * epoch_num_per_var,
                               min(self.__epochs_num, (param_idx + 1) * epoch_num_per_var)):
                self.__train_one_epoch(epoch=epoch, correction_func=self.__correction(centering=True, rotation=True))

    def __train_one_epoch(self, epoch, correction_func=None):

        self.train()

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
            print(f"| Epoch = {epoch} | Loss/train: {average_epoch_loss} |")

    def forward(self, time_seq_list, node_pairs):

        return self.get_negative_log_likelihood(time_seq_list, node_pairs) + \
               self.__current_reg_lambda * self.__regularization_func(node_idx=torch.unique(node_pairs))

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None, bins_rwidth_grad=None, reg_params_grad=None):

        if beta_grad is not None:
            self._beta.requires_grad = beta_grad

        if x0_grad is not None:
            self._x0.requires_grad = x0_grad

        if v_grad is not None:
            self._v.requires_grad = v_grad
            self.__current_reg_lambda = self.__reg_lambda

            if self.__reg_kron:

                self.__reg_sigma.requires_grad = True
                self.__reg_BL.requires_grad = True
                self.__reg_Cd.requires_grad = True
                # self.__reg_CV.requires_grad = True
                self.__reg_CU.requires_grad = True
                self.__reg_CR.requires_grad = True

            else:

                self.__reg_c.requires_grad = True
                self.__reg_sigma.requires_grad = True



        if bins_rwidth_grad is not None:
            self._bins_rwidth.requires_grad = bins_rwidth_grad

        if reg_params_grad is not None:
            self.__reg_params.requires_grad = reg_params_grad

    def __correction(self, centering=None, rotation=None):

        with torch.no_grad():

            if centering:
                x0_m = torch.mean(self._x0, dim=0, keepdim=True)
                self._x0 -= x0_m

                v_m = torch.mean(self._v, dim=1, keepdim=True)
                self._v -= v_m

            if rotation:
                border = self._x0.shape[0]
                U, S, _ = torch.linalg.svd(
                    torch.vstack((self._x0, self._v.view(-1, self._v.shape[2]))),
                    full_matrices=False
                )

                temp = U @ torch.diag_embed(S)

                self._x0.data = temp[:border, :]
                self._v.data = temp[border:, :].view(self._v.shape[0], self._v.shape[1], self._v.shape[2])

    def __get_gp_rbf_kernel(self, sigma_square, cholesky=True):

        # Get the number of bin size
        bin_num = self.get_num_of_bins()

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, bin_num) / 2.

        # Compute the inverse of kernel/covariance matrix
        time_mat = ((middle_bounds - middle_bounds.transpose(1, 2)) ** 2)


        if len(sigma_square) > 1:
            time_mat = time_mat.expand(self._dim, bin_num, bin_num)
        else:
            time_mat = time_mat.squeeze(0)

        kernel = torch.exp(-0.5 * torch.div(time_mat, sigma_square))
        if cholesky:
            inv_kernel = torch.linalg.cholesky(torch.cholesky_inverse(kernel))
        else:
            inv_kernel = torch.linalg.inv(kernel)

        return torch.logdet(kernel), inv_kernel

    def __neg_log_gp_regularization(self, node_idx, cholesky=True):

        # Compute the inverse of kernel/covariance matrix
        sigma_square = torch.mul(self.__reg_sigma, self.__reg_sigma).unsqueeze(0).view(self._dim, 1, 1)
        log_det_kernel, inv_kernel = self.__get_gp_rbf_kernel(sigma_square, cholesky=cholesky)

        # Multiply it by a scaling coefficients
        lambda_square = torch.mul(self.__reg_c, self.__reg_c).view(self._dim, 1, 1)
        inv_kernel = torch.mul(lambda_square, inv_kernel)

        # v: B x N x D tensor
        chosen_v = self._v.transpose(0, 2)[:, node_idx, :]
        # chosen_v: D x chosen_N x B
        # inv_kernel: D x B x B
        # log_exp_term: D x chosen_N -> scalar
        log_exp_term = -torch.bmm(
            chosen_v, inv_kernel
        ).mul(chosen_v).sum(dim=2, keepdim=False).sum()
        # log_det: D vector
        log_det = -0.5 * self.get_num_of_bins() + log_det_kernel.sum() * len(node_idx)
        log_pi = -0.5 * torch.log(2 * torch.as_tensor(math.pi)) * len(node_idx) * self._dim

        return -(log_pi + log_det + log_exp_term)

    def __neg_log_kronecker_gp_regularization(self, node_idx, cholesky=True):

        # A and inv_A: T x T matrix
        sigma_square = self.__reg_sigma * self.__reg_sigma
        log_det_A, inv_A = self.__get_gp_rbf_kernel(sigma_square, cholesky=cholesky)

        # inv_B: D x D matrix -> inv_BL: DxD lower triangular matrix
        inv_B_L = torch.inverse(self.__reg_BL)
        inv_B = torch.matmul(inv_B_L.transpose(0, 1), inv_B_L)

        # inv_C: N x N matrix
        # d = self.__reg_Cd[node_idx]
        d = torch.ones(size=(len(node_idx), ), dtype=torch.float)

        V = torch.mm(torch.diag(torch.div(1.0, d)), torch.mm(self.__reg_CU[node_idx, :], self.__reg_CR))  #self.__reg_CV[node_idx, :]
        inv_C = torch.diag(torch.div(1.0, d)) - torch.mm(V, V.transpose(0, 1))

        log_det_kernel = + (self._dim * len(node_idx)) * log_det_A \
                         - (self.get_num_of_bins()*len(node_idx)) * torch.logdet(inv_B) \
                         - (self.get_num_of_bins()*self._dim) * torch.logdet(inv_C)

        inv_kernel = torch.kron(inv_A.contiguous(), torch.kron(inv_B, inv_C))

        # v: B x N x D tensor
        # chosen: B x chosen_N x D tensor ->
        chosen = self._v.transpose(1, 2)[:, :, node_idx].flatten()

        result = -0.5 * self._dim * len(node_idx) * self.get_num_of_bins() * ( log_det_kernel + 2 * math.pi) \
                 - torch.mul(torch.mul(chosen, inv_kernel), chosen)

        return -result.sum()