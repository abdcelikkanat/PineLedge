import random
import numpy as np
import torch
from utils import const
from utils import mean_normalization
from torch.nn.functional import pdist
import time


class BaseModel(torch.nn.Module):
    '''
    Description
    '''
    def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, last_time: float,
                 bins_rwidth: int or torch.Tensor = None, device: torch.device = "cpu", verbose: bool = False, seed: int = 0):
        '''

        :param x0: initial position tensor of size N x D
        :param v: velocity tensor of size I x N X D where I is the number of intervals/bins
        :param beta: bias terms for the nodes, a tensor of size N
        :param bins_rwidth: # Relative bin width sizes of the timeline
        :param seed: seed value
        '''

        super(BaseModel, self).__init__()

        self._x0 = x0
        self._v = v
        self._beta = beta
        self._seed = seed
        self._init_time = 0  # It is always assumed that the initial time is 0
        self._last_time = last_time
        self._bins_rwidth = torch.ones(size=(bins_rwidth,), dtype=torch.float, device=device) / bins_rwidth if type(bins_rwidth) is int else torch.as_tensor(bins_rwidth, device=device)
        self._verbose = verbose
        self._device = device

        # Extract the number of nodes and the dimension size
        self._nodes_num = self._x0.shape[0]
        self._dim = self._x0.shape[1]

        # Check if the given parameters have correct shapes
        self._check_input_params()

        # Set the seed value for reproducibility
        self._set_seed()

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

    def get_num_of_bins(self):

        return len(self._bins_rwidth)

    def get_bins_bounds(self, bins_rwidth=None):

        if bins_rwidth is None:
            bins_rwidth = self._bins_rwidth.type(torch.float)

        timeline_len = self._last_time - self._init_time
        bounds = torch.cat((torch.as_tensor([self._init_time], device=self._device),
                            self._init_time + torch.cumsum(torch.softmax(bins_rwidth, dim=0), dim=0) * timeline_len
                            ))

        return bounds

    def get_bins_widths(self, bounds=None):

        if bounds is None:
            bounds = self.get_bins_bounds()

        return bounds[1:] - bounds[:-1]

    # def compute_bin_bound_width(self):
    #     self._bin_boundaries = torch.cat((torch.zeros(1), self._last_time * torch.cumsum(torch.softmax(self._gamma, dim=0), dim=0)))
    #     self._bins_width = self._bin_boundaries[1:] - self._bin_boundaries[:-1]
    #     return

    # def get_bin_boundaries(self):
    #
    #     return self._bin_boundaries

    def get_xt(self, times_list: torch.Tensor, x0: torch.Tensor = None, v: torch.Tensor = None, bin_bounds: torch.Tensor = None) -> torch.Tensor:
        # Compute the positions of nodes at time t
        if x0 is None:
            x0 = mean_normalization(self._x0)
        if v is None:
            v = mean_normalization(self._v)
        if bin_bounds is None:
            # Get the bin boundaries and widths
            bin_bounds = self.get_bins_bounds()

        # if len(x0.shape) == 1:
        #     x0 = x0.view(1, self._dim)
        #     v = v.view(self.get_num_of_bins(), 1, self._dim)

        bin_widths = self.get_bins_widths(bounds=bin_bounds) # ---> Bin widths are always same so fix it to make it faster

        # Find the interval index that the given time point lies on
        boundaries = bin_bounds[1:-1] if len(bin_bounds) > 2 else torch.as_tensor([bin_bounds[1]+1e-6])
        time_interval_indices = torch.bucketize(times_list, boundaries=boundaries, right=True)

        # Compute the distance of the time points to the initial time of the intervals that they lay on
        time_remainders = times_list - bin_bounds[time_interval_indices]

        # Construct a matrix storing the time differences (delta_t)
        # Compute the total displacements for each time-intervals and append a zero column to get rid of indexing issues
        cum_displacement = torch.cat((
            torch.zeros(1, x0.shape[0], x0.shape[1], device=self._device),
            torch.cumsum(torch.mul(v, bin_widths[:, None, None]), dim=0)
        ))
        xt = x0 + cum_displacement[time_interval_indices, :, :]
        # Finally, add the the displacement on the interval that nodes lay on
        xt += torch.mul(v[time_interval_indices, :, :], time_remainders[:, None, None])

        return xt

    def get_norm_sum(self, alpha, xt_bins, time_indices, bin_bounds,
                     times_list: torch.Tensor, distance: str = "squared_euc") -> torch.Tensor:

        if distance != "squared_euc":
            raise NotImplementedError("Not implemented for other than squared euclidean.")

        p = torch.sigmoid(alpha).view(-1, 1).expand(self.get_num_of_bins(), self._dim)
        bin_counts = torch.bincount(time_indices, minlength=self.get_num_of_bins()).type(torch.float)
        # print(":> ", xt_bins.shape, p.shape, )
        squared_norm = torch.norm(
            (1 - p) * xt_bins[:-1, :] + p * xt_bins[1:, :], dim=1, keepdim=False
        ) ** 2
        # print(":>", self.get_num_of_bins(), bin_counts.shape, squared_norm.shape)
        return torch.dot(bin_counts, squared_norm)

        # # temp is a tensor of size len(times_list) x node_pairs.shape[0] x dim
        # temp = self.get_xt(times_list=times_list, x0=x0, v=v)
        #
        # return torch.sum(torch.norm(temp, p=2, dim=2, keepdim=False) ** 2)

    def get_pairwise_distances(self, times_list: torch.Tensor, node_pairs: torch.Tensor = None,
                               distance: str = "squared_euc"):

        if node_pairs is None:

            raise NotImplementedError("It should be implemented for every node pairs!")

        else:
            x_tilde = mean_normalization(self._x0)
            v_tilde = mean_normalization(self._v)

            delta_x0 = x_tilde[node_pairs[0], :] - x_tilde[node_pairs[1], :]
            delta_v = v_tilde[:, node_pairs[0], :] - v_tilde[:, node_pairs[1], :]

            # temp is a tensor of size len(times_list) x node_pairs.shape[0] x dim
            temp = self.get_xt(times_list=times_list, x0=delta_x0, v=delta_v)

            if distance == "squared_euc":
                norm = torch.norm(temp, p=2, dim=2, keepdim=False) ** 2
            elif distance == "euc":
                norm = torch.norm(temp, p=2, dim=2, keepdim=False)
            else:
                raise ValueError("Invalid distance metric!")

            return norm

    def get_intensity(self, times_list: torch.tensor, node_pairs: torch.tensor, distance: str = "squared_euc"):

        # Add an additional axis for beta parameters for time dimension
        intensities = -self.get_pairwise_distances(times_list=times_list, node_pairs=node_pairs, distance=distance)
        # print("o: ", times_list, intensities.shape)
        # intensities += (self._beta[node_pairs[0]] + self._beta[node_pairs[1]]).unsqueeze(0)
        # time_indices = torch.bucketize(times_list, boundaries=self.get_bins_bounds()[1:-1], right=True)
        intensities += (self._beta**2).expand( len(node_pairs[0]) )  #intensities += (self._beta[node_pairs[0]] + self._beta[node_pairs[1]])

        return torch.exp(intensities)

    def get_log_intensity(self, times_list: torch.Tensor, node_pairs: torch.Tensor, distance: str = "squared_euc"):

        # Add an additional axis for beta parameters for time dimension
        intensities = -self.get_pairwise_distances(times_list=times_list, node_pairs=node_pairs, distance=distance)
        intensities += (self._beta**2).expand(len(times_list), node_pairs.shape[1]) #intensities += (self._beta[node_pairs[0]] + self._beta[node_pairs[1]]).expand(len(times_list), node_pairs.shape[1])

        return intensities

    def get_intensity_integral(self, x0: torch.Tensor = None, v: torch.Tensor = None, beta: torch.Tensor = None, bin_bounds: torch.Tensor = None,
                               node_pairs: torch.tensor = None, distance: str = "squared_euc"):

        if x0 is None or v is None or bin_bounds is None:
            x0 = mean_normalization(self._x0)
            v = mean_normalization(self._v)
            bin_bounds = self.get_bins_bounds()
            # raise NotImplementedError("Not implemented for given x0 and v!")

        if beta is None:
            beta = self._beta

        if node_pairs is None:
            node_pairs = torch.triu_indices(self._nodes_num, self._nodes_num, offset=1)

        # Common variables
        delta_x0 = x0[node_pairs[0], :] - x0[node_pairs[1], :]
        delta_v = v[:, node_pairs[0], :] - v[:, node_pairs[1], :]
        beta_ij = (beta**2).expand(len(node_pairs[0]))  #beta[node_pairs[0]] + beta[node_pairs[1]]

        # if len(node_pairs.shape) == 1:
        #     delta_x0 = delta_x0.view(1, self._dim)
        #     delta_v = delta_v.view(self.get_num_of_bins(), 1, self._dim)

        if distance == "squared_euc":

            delta_xt = self.get_xt(times_list=bin_bounds[:-1], x0=delta_x0, v=delta_v, bin_bounds=bin_bounds)

            norm_delta_xt = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
            # norm_v: a matrix of bins_counts x len(node_pairs)
            norm_delta_v = torch.norm(delta_v, p=2, dim=2, keepdim=False)
            inv_norm_delta_v = 1.0 / (norm_delta_v + const.eps)
            delta_xt_v = (delta_xt * delta_v).sum(dim=2, keepdim=False)
            r = delta_xt_v * inv_norm_delta_v

            term0 = 0.5 * torch.sqrt(const.pi).to(self._device) * inv_norm_delta_v
            # term1 = torch.exp( beta_ij.unsqueeze(0) + r**2 - norm_delta_xt**2 )
            term1 = torch.exp(beta_ij.unsqueeze(0) + r ** 2 - norm_delta_xt ** 2)
            term2_u = torch.erf(bin_bounds[1:].expand(norm_delta_v.shape[1], len(bin_bounds)-1).t()*norm_delta_v + r)
            term2_l = torch.erf(bin_bounds[:-1].expand(norm_delta_v.shape[1], len(bin_bounds)-1).t()*norm_delta_v + r)

            # From bins_counts x len(node_pairs) matrix to a vector
            return (term0 * term1 * (term2_u - term2_l)).sum(dim=0)

        # elif distance == "euc":
        #
        #     delta_xt = self.get_xt(times_list=bin_bounds, x0=delta_x0, v=delta_v, bin_bounds=bin_bounds)
        #
        #     delta_xt_norm = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
        #     delta_exp_term = torch.exp(
        #         beta_ij.unsqueeze(0) - delta_xt_norm
        #     )
        #
        #     numer = torch.mul(delta_xt_norm, delta_exp_term)
        #     term1 = torch.divide(numer[1:, :], torch.mul(delta_xt[1:, :, :], delta_v).sum(dim=2) + const.eps)
        #     term0 = torch.divide(numer[:-1, :], torch.mul(delta_xt[:-1, :, :], delta_v).sum(dim=2) + const.eps)
        #
        #     # term1 - term0 is a matrix of size bins_counts x len(node_pairs)
        #     return torch.sum(term1 - term0, dim=0)

        else:

            raise ValueError("Invalid distance metric!")

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
            inv_norm_delta_v = 1.0 / (norm_delta_v + const.eps)
            # print(delta_xt.shape, delta_v[time_indices, :, :].shape)
            delta_xt_v = (delta_xt * delta_v.index_select(0, time_indices)).sum(dim=2, keepdim=False)
            r = delta_xt_v * inv_norm_delta_v

            term0 = 0.5 * torch.sqrt(const.pi).to(self._device) * inv_norm_delta_v
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

    def get_negative_log_likelihood(self, time_seq_list: list, node_pairs: torch.tensor):

        it = time.time()
        nll = 0
        integral_term = -self.get_intensity_integral(node_pairs=node_pairs).sum()

        it = time.time()
        non_integral_term = 0
        for idx in range(node_pairs.shape[1]):

            nodes_pair = node_pairs[:, idx].view(2, 1)
            times_list = time_seq_list[idx]

            non_integral_term += torch.sum(self.get_log_intensity(times_list=torch.as_tensor(times_list, device=self._device), node_pairs=nodes_pair))

        return -(integral_term + non_integral_term)


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

        return self._x0

    def get_v(self):

        return self._v

    def get_dim(self):

        return self._dim

    def get_number_of_nodes(self):

        return self._nodes_num


