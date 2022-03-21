import torch

def get_rbf_kernel(self, time_mat):
    kernel = self.__prior_rbf_sigma ** 2 * torch.exp(
        -0.5 * torch.div(time_mat ** 2, self.__prior_rbf_l ** 2)
    )

    return kernel


def get_periodic_kernel(self, time_mat):
    kernel = self.__prior_periodic_sigma ** 2 * torch.exp(
        -2 * torch.sin(math.pi * torch.abs(time_mat) / self.__prior_periodic_p) ** 2 / (self.__prior_periodic_l ** 2)
    )

    return kernel


def __get_A(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor,
            get_inv: bool = True, cholesky: bool = True):

    # Compute the inverse of kernel/covariance matrix
    time_mat = bin_centers1 - bin_centers2.transpose(1, 2)
    time_mat = time_mat.squeeze(0)

    kernel = 0.
    for kernel_name in self.__prior_A_kernel_names:
        kernel_func = getattr(self, 'get_' + kernel_name + '_kernel')
        kernel += kernel_func(time_mat=time_mat)

    # Add a noise term
    kernel += torch.eye(n=kernel.shape[0], m=kernel.shape[1]) * (self.__prior_kernel_noise_sigma ** 2)

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

    # B: D x D matrix -> self._prior_B_L: D x D lower triangular matrix
    B_L = torch.tril(self.__prior_B_L, diagonal=-1) + torch.diag(torch.diag(self.__prior_B_L) ** 2)
    inv_B = torch.cholesky_inverse(B_L)

    # C: N x N matrix
    C_Q = self.__prior_C_Q[nodes, :]

    C_inv_D = torch.eye(n=len(nodes))  # torch.diag(1.0 / self.__prior_C_lambda.expand(len(nodes)))
    # C = C_D + (C_Q @ C_Q.t())
    # By Woodbury matrix identity
    invDQ = C_inv_D @ C_Q
    inv_C = C_inv_D - invDQ @ torch.inverse(torch.eye(C_Q.shape[1]) + C_Q.t() @ C_inv_D @ C_Q) @ invDQ.t()

    # Compute the product in an efficient way
    # v^t ( A kron B kron C ) v
    batch_v = self._v[:, nodes, :]
    # Vectorize the velocity tensor
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