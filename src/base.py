import random
import numpy as np
import torch
from utils import *


class BaseModel(torch.nn.Module):
    '''
    Description
    '''
    def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, bins_num: int = 10,
                 last_time: float = 1.0, prior_lambda: float = 1e5, prior_sigma: torch.Tensor = None,
                 prior_B_sigma: torch.Tensor = None, prior_C_Q: torch.Tensor= None,
                 node_pairs_mask: torch.Tensor = None, device: torch.device = "cpu",
                 verbose: bool = False, seed: int = 0):

        super(BaseModel, self).__init__()

        self._x0 = x0
        self._v = v
        self._beta = beta
        self._seed = seed
        self._init_time = 0  # It is always assumed that the initial time is 0
        self._last_time = last_time
        self._bins_num = bins_num
        self._bin_width = (self._last_time - self._init_time) / float(self._bins_num)
        self._verbose = verbose
        self._device = device

        # Extract the number of nodes and the dimension size
        self._nodes_num = self._x0.shape[0]
        self._dim = self._x0.shape[1]

        # Initialize the parameters of prior function
        # scaling factor of covariance matrix
        self._prior_lambda = torch.as_tensor(prior_lambda, dtype=torch.float, device=self._device)
        # noise deviation
        self._prior_sigma = torch.as_tensor(prior_sigma, dtype=torch.float, device=self._device)
        # length-scale parameter of RBF kernel used in the construction of B
        self._prior_B_sigma = torch.as_tensor(prior_B_sigma, dtype=torch.float, device=self._device)
        self._prior_C_Q = prior_C_Q  # the parameter required for the construction of the matrix C
        self.__R, self.__R_factor, self.__R_factor_inv = None, None, None  # Capacitance matrix

        # Check if the given parameters have correct shapes
        self._check_input_params()

        # Set the seed value for reproducibility
        self._set_seed()

        self.__node_pairs_mask = node_pairs_mask

    def _check_input_params(self):

        assert self._nodes_num == self._v.shape[1], \
            "The initial position, velocity and bias tensors must contain the same number of nodes."

        assert self._dim == self._v.shape[2], \
            "The dimensions of the initial position and velocity tensors must be the same."

        return 0

    def _set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

    def get_bins_bounds(self):

        bounds = self._init_time + torch.cat((
            torch.as_tensor([self._init_time], device=self._device),
            torch.cumsum(torch.ones(self._bins_num, device=self._device) * self._bin_width, dim=0)
        ))

        return bounds

    def get_xt(self, events_times_list: torch.Tensor, x0: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:

        if x0 is None or v is None:
            raise ValueError("x0 and v cannot be none!")
            # x0 = torch.repeat_interleave(mean_normalization(self._x0), repeats=len(events_times_list), dim=0)
            # v = torch.repeat_interleave(mean_normalization(self._v), repeats=len(events_times_list), dim=1)
            # events_times_list = events_times_list.repeat(self._nodes_num)

        assert len(events_times_list) == x0.shape[0] and x0.shape[0] == v.shape[1], print( len(events_times_list), x0.shape, v.shape)

        # events_times_list[events_times_list == self._last_time] = self._last_time - utils.EPS
        # Compute the event indices and residual times
        events_bin_indices = torch.div(events_times_list, self._bin_width, rounding_mode='floor').type(torch.int)
        residual_time = events_times_list % self._bin_width
        events_bin_indices[events_bin_indices == self._bins_num] = self._bins_num - 1

        # Compute the total displacements for each time-intervals and append a zero column to get rid of indexing issues
        cum_displacement = torch.cat((
            torch.zeros(1, x0.shape[0], self._dim, device=self._device),
            torch.cumsum(v * self._bin_width, dim=0)
        )).view(-1, self._dim)

        xt = x0 + torch.index_select(cum_displacement, dim=0, index=events_bin_indices * v.shape[1] + torch.arange(len(events_bin_indices)))
        # Finally, add the the displacement on the interval that nodes lay on
        xt = xt + torch.mul(
            residual_time.unsqueeze(1),
            torch.index_select(v.view(-1, self._dim), dim=0, index=events_bin_indices * v.shape[1] + torch.arange(len(events_bin_indices)))
        )

        return xt

    def get_pairwise_distances(self, times_list: torch.Tensor, node_pairs: torch.Tensor = None,
                               distance: str = "squared_euc"):

        if node_pairs is None:
            raise NotImplementedError("It should be implemented for every node pairs!")

        if distance != "squared_euc":
            raise ValueError("Invalid distance metric!")

        x_tilde = mean_normalization(self._x0)
        v_tilde = mean_normalization(self._v)

        delta_x0 = torch.index_select(x_tilde, dim=0, index=node_pairs[0]) - \
                   torch.index_select(x_tilde, dim=0, index=node_pairs[1])
        delta_v = torch.index_select(v_tilde, dim=1, index=node_pairs[0]) - \
                  torch.index_select(v_tilde, dim=1, index=node_pairs[1])

        # delta_xt is a tensor of size len(times_list) x dim
        delta_xt = self.get_xt(events_times_list=times_list, x0=delta_x0, v=delta_v)
        # Compute the squared Euclidean distance
        norm = torch.norm(delta_xt, p=2, dim=1, keepdim=False) ** 2

        return norm

    def get_intensity(self, times_list: torch.tensor, node_pairs: torch.tensor, distance: str = "squared_euc"):

        return torch.exp(self.get_log_intensity(times_list, node_pairs, distance))

    def get_log_intensity(self, times_list: torch.Tensor, node_pairs: torch.Tensor, distance: str = "squared_euc"):

        # Mask given node pairs
        if self.__node_pairs_mask is not None:
            non_idx = torch.cdist(node_pairs.T.float(), self.__node_pairs_mask.T.float()).nonzero(as_tuple=True)[0]
            idx = torch.unique(non_idx, sorted=True)
            node_pairs = node_pairs[:, idx]

        # Get pairwise distances
        intensities = -self.get_pairwise_distances(times_list=times_list, node_pairs=node_pairs, distance=distance)
        # Add an additional axis for beta parameters for time dimension
        intensities += torch.index_select(self._beta, dim=0, index=node_pairs[0]) + \
                       torch.index_select(self._beta, dim=0, index=node_pairs[1])

        return intensities

    def get_intensity_integral(self, nodes: torch.tensor, x0: torch.Tensor = None, v: torch.Tensor = None,
                               beta: torch.Tensor = None, bin_bounds: torch.Tensor = None,
                               distance: str = "squared_euc"):

        if x0 is None or v is None:
            x0 = mean_normalization(self._x0)
            v = mean_normalization(self._v)

        if bin_bounds is None:
            bin_bounds = self.get_bins_bounds()

        if beta is None:
            beta = self._beta

        batch_size = len(nodes)
        unique_node_pairs = torch.as_tensor(
            [[nodes[i], nodes[j]] for i in range(batch_size) for j in range(i+1, batch_size)],
            dtype=torch.int, device=self._device
        ).t()
        # Mask given node pairs
        if self.__node_pairs_mask is not None:
            non_idx = torch.cdist(unique_node_pairs.T.float(), self.__node_pairs_mask.T.float()).nonzero(as_tuple=True)[0]
            idx = torch.unique(non_idx, sorted=True)
            unique_node_pairs = unique_node_pairs[:, idx]

        # Common variables
        delta_x0 = torch.index_select(x0, dim=0, index=unique_node_pairs[0]) - \
                   torch.index_select(x0, dim=0, index=unique_node_pairs[1])
        delta_v = torch.index_select(v, dim=1, index=unique_node_pairs[0]) - \
                  torch.index_select(v, dim=1, index=unique_node_pairs[1])
        beta_ij = torch.index_select(beta, dim=0, index=unique_node_pairs[0]) + \
                  torch.index_select(beta, dim=0, index=unique_node_pairs[1])

        # if len(node_pairs.shape) == 1:
        #     delta_x0 = delta_x0.view(1, self._dim)
        #     delta_v = delta_v.view(self.get_num_of_bins(), 1, self._dim)

        if distance != "squared_euc":
            raise ValueError("Invalid distance metric!")

        delta_xt = self.get_xt(
            events_times_list=torch.cat([bin_bounds[:-1]] * delta_x0.shape[0]),
            x0=torch.repeat_interleave(delta_x0, repeats=len(bin_bounds)-1, dim=0),
            v=torch.repeat_interleave(delta_v, repeats=len(bin_bounds)-1, dim=1),
        ).reshape((delta_x0.shape[0], len(bin_bounds)-1,  self._dim)).transpose(0, 1)

        norm_delta_xt = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
        # norm_v: a matrix of bins_counts x len(node_pairs)
        norm_delta_v = torch.norm(delta_v, p=2, dim=2, keepdim=False)
        inv_norm_delta_v = 1.0 / (norm_delta_v + utils.EPS)
        delta_xt_v = (delta_xt * delta_v).sum(dim=2, keepdim=False)
        r = delta_xt_v * inv_norm_delta_v

        term0 = 0.5 * torch.sqrt(torch.as_tensor(utils.PI, device=self._device)).to(self._device) * inv_norm_delta_v
        # term1 = torch.exp( beta_ij.unsqueeze(0) + r**2 - norm_delta_xt**2 )
        term1 = torch.exp(beta_ij.unsqueeze(0) + r ** 2 - norm_delta_xt ** 2)
        term2_u = torch.erf(bin_bounds[1:].expand(norm_delta_v.shape[1], len(bin_bounds)-1).t()*norm_delta_v + r)
        term2_l = torch.erf(bin_bounds[:-1].expand(norm_delta_v.shape[1], len(bin_bounds)-1).t()*norm_delta_v + r)

        # From bins_counts x len(node_pairs) matrix to a vector
        return (term0 * term1 * (term2_u - term2_l)).sum(dim=0)

    def get_intensity_integral_for(self, i: int, j: int, interval: torch.Tensor = None,
                               distance: str = "squared_euc"):

        x0 = mean_normalization(self._x0)
        v = mean_normalization(self._v)
        beta = self._beta

        unique_node_pairs = torch.as_tensor([i, j], dtype=torch.int, device=self._device).t()

        # Mask given node pairs
        if self.__node_pairs_mask is not None:
            non_idx = torch.cdist(unique_node_pairs.T.float(), self.__node_pairs_mask.T.float()).nonzero(as_tuple=True)[0]
            idx = torch.unique(non_idx, sorted=True)
            unique_node_pairs = unique_node_pairs[:, idx]

        # Common variables
        delta_x0 = torch.index_select(x0, dim=0, index=unique_node_pairs[0]) - \
                   torch.index_select(x0, dim=0, index=unique_node_pairs[1])
        delta_v = torch.index_select(v, dim=1, index=unique_node_pairs[0]) - \
                  torch.index_select(v, dim=1, index=unique_node_pairs[1])
        beta_ij = torch.index_select(beta, dim=0, index=unique_node_pairs[0]) + \
                  torch.index_select(beta, dim=0, index=unique_node_pairs[1])

        # if len(node_pairs.shape) == 1:
        #     delta_x0 = delta_x0.view(1, self._dim)
        #     delta_v = delta_v.view(self.get_num_of_bins(), 1, self._dim)

        if distance != "squared_euc":
            raise ValueError("Invalid distance metric!")

        delta_xt = self.get_xt(
            events_times_list=torch.cat([interval[:-1]] * delta_x0.shape[0]),
            x0=torch.repeat_interleave(delta_x0, repeats=len(interval)-1, dim=0),
            v=torch.repeat_interleave(delta_v, repeats=len(interval)-1, dim=1),
        ).reshape((delta_x0.shape[0], len(interval)-1,  self._dim)).transpose(0, 1)

        norm_delta_xt = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
        # norm_v: a matrix of bins_counts x len(node_pairs)
        norm_delta_v = torch.norm(delta_v, p=2, dim=2, keepdim=False)
        inv_norm_delta_v = 1.0 / (norm_delta_v + utils.EPS)
        delta_xt_v = (delta_xt * delta_v).sum(dim=2, keepdim=False)
        r = delta_xt_v * inv_norm_delta_v

        term0 = 0.5 * torch.sqrt(torch.as_tensor(utils.PI, device=self._device)).to(self._device) * inv_norm_delta_v
        # term1 = torch.exp( beta_ij.unsqueeze(0) + r**2 - norm_delta_xt**2 )
        term1 = torch.exp(beta_ij.unsqueeze(0) + r ** 2 - norm_delta_xt ** 2)
        term2_u = torch.erf(interval[1:].expand(norm_delta_v.shape[1], len(interval)-1).t()*norm_delta_v + r)
        term2_l = torch.erf(interval[:-1].expand(norm_delta_v.shape[1], len(interval)-1).t()*norm_delta_v + r)

        # From bins_counts x len(node_pairs) matrix to a vector
        return (term0 * term1 * (term2_u - term2_l)).sum(dim=0)

    def get_survival_function(self, times_list: torch.Tensor, x0: torch.Tensor = None, v: torch.Tensor = None,
                              beta: torch.Tensor = None, bin_bounds: torch.Tensor = None,
                              node_pairs: torch.tensor = None, distance: str = "squared_euc"):

        if x0 is None or v is None or bin_bounds is None:
            x0 = mean_normalization(self._x0)
            v = mean_normalization(self._v)
            bin_bounds = self.get_bins_bounds()

        if beta is None:
            beta = self._beta

        if node_pairs is None:
            node_pairs = torch.triu_indices(self._nodes_num, self._nodes_num, offset=1)

        # Common variables
        delta_x0 = x0[node_pairs[0], :] - x0[node_pairs[1], :]
        delta_v = v[:, node_pairs[0], :] - v[:, node_pairs[1], :]
        beta_ij = (beta**2).expand(len(node_pairs[0]))

        all_indices = torch.as_tensor([0] * len(bin_bounds) + [1] * len(times_list), dtype=torch.int)
        all_bounds = torch.hstack((bin_bounds, times_list))
        all_bounds, sorting_idx = torch.sort(all_bounds)
        sorted_indices = all_indices[sorting_idx]
        all_indices = torch.cumsum(sorted_indices, dim=0)
        all_indices[all_indices % 2 == 0] = 0
        # all_indices[sorted_indices] = 1
        # all_indices = 1 - all_indices
        all_indices = (all_indices == 0).nonzero()

        # Time indices
        time_indices = torch.bucketize(all_bounds[:-1], boundaries=bin_bounds[1:-1], right=True)

        if distance == "squared_euc":

            delta_xt = self.get_xt(times_list=all_bounds[:-1], x0=delta_x0, v=delta_v, bin_bounds=bin_bounds)

            norm_delta_xt = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
            # norm_v: a matrix of bins_counts x len(node_pairs)
            norm_delta_v = torch.norm(delta_v.index_select(0, time_indices), p=2, dim=2, keepdim=False)
            inv_norm_delta_v = 1.0 / (norm_delta_v + utils.EPS)
            # print(delta_xt.shape, delta_v[time_indices, :, :].shape)
            delta_xt_v = (delta_xt * delta_v.index_select(0, time_indices)).sum(dim=2, keepdim=False)
            r = delta_xt_v * inv_norm_delta_v

            term0 = 0.5 * torch.sqrt(utils.PI).to(self._device) * inv_norm_delta_v
            # term1 = torch.exp( beta_ij.unsqueeze(0) + r**2 - norm_delta_xt**2 )
            term1 = torch.exp(beta_ij.unsqueeze(0) + r ** 2 - norm_delta_xt ** 2)
            term2_u = torch.erf(all_bounds[1:].expand(norm_delta_v.shape[1], len(all_bounds)-1).t()*norm_delta_v + r)
            term2_l = torch.erf(all_bounds[:-1].expand(norm_delta_v.shape[1], len(all_bounds)-1).t()*norm_delta_v + r)

            # There might be more efficient implementation!
            diff = term2_u - term2_l
            diff = diff[all_indices[:-1]]

            return (term0 * term1 * diff).sum(dim=0)

        else:

            raise ValueError("Invalid distance metric!")

    def get_negative_log_likelihood(self, nodes: torch.Tensor, event_times: torch.Tensor, event_node_pairs: torch.Tensor):

        integral_term = -self.get_intensity_integral(nodes=nodes).sum()

        non_integral_term = self.get_log_intensity(times_list=event_times, node_pairs=event_node_pairs).sum()

        return -( non_integral_term + integral_term)

    def get_survival_log_likelihood(self, time_seq_list: list, node_pairs: torch.tensor):

        integral_term = 0
        non_integral_term = 0
        for idx in range(node_pairs.shape[1]):

            times_list = time_seq_list[idx]
            node_pair = node_pairs[:, idx]

            # if len(times_list):

            integral_term += -self.get_survival_function(
                times_list=torch.as_tensor(times_list, device=self._device), node_pairs=torch.as_tensor(node_pair).unsqueeze(1)
            ).sum()

            non_integral_term += torch.sum(
                self.get_log_intensity(times_list=torch.as_tensor(times_list, device=self._device),
                                       node_pairs=torch.as_tensor(node_pair).unsqueeze(1)))

        return -(integral_term + non_integral_term)

    def get_x0(self):

        return mean_normalization(self._x0)

    def get_v(self):

        return mean_normalization(self._v)

    def get_dim(self):

        return self._dim

    def get_number_of_nodes(self):

        return self._nodes_num

    def _get_B_factor(self, bin_centers1: torch.Tensor, bin_centers2: torch.Tensor, only_kernel=False):
        time_mat = bin_centers1 - bin_centers2.transpose(1, 2)
        time_mat = time_mat.squeeze(0)

        B_sigma = utils.softplus(self._prior_B_sigma)
        kernel = torch.exp(-0.5 * torch.div(time_mat ** 2, B_sigma))

        # Add a constant term to get rid of computational problems
        kernel = kernel + 10*utils.EPS * torch.eye(n=kernel.shape[0], m=kernel.shape[1])

        if only_kernel:
            return kernel

        # B x B lower triangular matrix
        L = torch.linalg.cholesky(kernel)

        return L

    def _get_C_factor(self):
        # K x N matrix
        return torch.softmax(self._prior_C_Q, dim=0)

    def _get_D_factor(self):
        # D x D matrix
        return torch.eye(self.get_dim())

    def get_neg_log_prior(self, batch_nodes, batch_num=0):

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, self._bins_num) / 2.

        # B x B matrix
        B_factor = self._get_B_factor(bin_centers1=middle_bounds, bin_centers2=middle_bounds)
        # N x K matrix where K is the community size
        C_factor = self._get_C_factor().T
        # D x D matrix
        D_factor = self._get_D_factor()

        # B(batch_size)D x BKD matrix
        K_factor_batch = torch.kron(
            B_factor.contiguous(),
            torch.kron(torch.index_select(C_factor.contiguous(), dim=0, index=batch_nodes), D_factor).contiguous()
        )

        # Some common parameters
        lambda_sq = self._prior_lambda ** 2
        sigma_sq = torch.sigmoid(self._prior_sigma)
        sigma_sq_inv = 1.0 / sigma_sq
        final_dim = self.get_number_of_nodes() * self._bins_num * self._dim
        reduced_dim = self._prior_C_Q.shape[0] * self._bins_num * self._dim

        # Compute the capacitance matrix R only if batch_num == 0
        if batch_num == 0:
            K_factor_full = torch.kron(B_factor.contiguous(), torch.kron(C_factor.contiguous(), D_factor).contiguous())
            self.__R = torch.eye(reduced_dim) + sigma_sq_inv * K_factor_full.T @ K_factor_full
            self.__R_factor = torch.linalg.cholesky(self.__R)
            self.__R_factor_inv = torch.inverse(self.__R)

        # Normalize and vectorize the velocities
        v_batch = mean_normalization(torch.index_select(self._v, dim=1, index=batch_nodes))
        v_vect_batch = utils.vectorize(v_batch).flatten()

        # Computation of the squared Mahalanobis distance: v.T @ inv(D + W @ W.T) @ v
        # It uses Woodbury matrix identity: inv(D + Kf @ Kf.T) = inv(D) - inv(D) @ Kf @ inv(R) @ Kf.T @ inv(D),
        # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
        mahalanobis_term1 = sigma_sq_inv * v_vect_batch.pow(2).sum(-1)
        mahalanobis_term2 = (sigma_sq_inv * v_vect_batch @ K_factor_batch @ self.__R_factor_inv.T).pow(2).sum(-1)
        m = (1.0 / lambda_sq) * (mahalanobis_term1 - mahalanobis_term2 )

        # Computation of the log determinant
        # It uses Matrix Determinant Lemma: log|D + Kf @ Kf.T| = log|R| + log|D|,
        # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
        log_det = 2 * self.__R_factor.diagonal(dim1=-2, dim2=-1).log().sum(-1) + final_dim * (lambda_sq.log() + sigma_sq.log())

        # Compute the negative log-likelihood
        log_prior_likelihood = -0.5 * (final_dim * utils.LOG2PI + log_det + m)

        return -log_prior_likelihood.squeeze(0)


