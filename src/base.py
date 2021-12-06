import random
import numpy as np
import torch
from torch.nn.functional import pdist


class BaseModel(torch.nn.Module):
    '''
    Description
    '''
    def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, last_time: float, bins_width: int or list = None, seed: int = 0):

        super(BaseModel, self).__init__()

        self._x0 = x0
        self._v = v
        self._beta = beta
        self._seed = seed
        self._init_time = 0  # It is always assumed that the initial time is 0
        self._bins_width = torch.tensor(bins_width)
        self._last_time = last_time

        self._bin_boundaries = None

        # Compute the bin boundaries

        if type(bins_width) is int:
            self._bin_boundaries = torch.linspace(self._init_time, self._last_time, steps=bins_width+1)
        else:
            bin_widths = torch.as_tensor(bins_width)
            bin_widths_cum_sum_ratios = torch.cumsum(bin_widths, dim=0) / torch.sum(bin_widths)
            self._bin_boundaries = (self._last_time - self._init_time) * bin_widths_cum_sum_ratios + self._init_time

        self._bins_width = self._bin_boundaries[1:] - self._bin_boundaries[:-1]

        self._bins_num = len(self._bin_boundaries) - 1

        # Extract the number of nodes and the dimension size
        self._nodes_num = self._x0.shape[0]
        self._dim = self._x0.shape[1]

        # Check if the given parameters have correct shapes
        assert self._nodes_num == self._v.shape[1] and self._nodes_num == self._beta.shape[0], \
            "The initial position, velocity and bias tensors must contain the same number of nodes."

        assert self._dim == self._v.shape[2], \
            "The dimensions of the initial position and velocity tensors must be the same."

        # Set the seed value for reproducibility
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

    def get_bin_boundaries(self):

        return self._bin_boundaries

    def get_bin_widths(self):

        return self._bins_width

    def get_xt(self, times_list: torch.Tensor, x0: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:

        if x0 is None:
            x0 = self._x0
        if v is None:
            v = self._v

        # Find the interval index that the given time point lies on
        time_interval_indices = torch.bucketize(times_list, boundaries=self._bin_boundaries[1:-1], right=True)

        # Compute the distance of the time points to the initial time of the intervals that they lay on
        time_remainders = times_list - self._bin_boundaries[time_interval_indices]

        # Construct a matrix storing the time differences (delta_t)
        # Compute the total displacements for each time-intervals and append a zero column to get rid of indexing issues
        cum_displacement = torch.cat((
            torch.zeros(1, x0.shape[0], x0.shape[1]),
            torch.cumsum(torch.mul(v, self._bins_width[:, None, None]), dim=0)
        ))
        xt = x0 + cum_displacement[time_interval_indices, :, :]
        # Finally, add the the displacement on the interval that nodes lay on
        xt += torch.mul(v[time_interval_indices, :, :], time_remainders[:, None, None])

        return xt

    def get_pairwise_distances(self, times_list: torch.Tensor, node_pairs: torch.Tensor = None):

        if node_pairs is None:

            raise NotImplementedError("It should be implemented for every node pairs!")
            # # Compute the pairwise distances for all node pairs
            # # xt is a tensor of size len(times_list) x num of nodes x dim
            # xt = self.get_xt(times_list=times_list)
            #
            # xt_norm = (xt**2).sum(dim=2)
            # xt_norm_view1 = xt_norm.view(xt.shape[0], xt.shape[1], 1)
            # xt_norm_view2 = xt_norm.view(xt.shape[0], 1, xt.shape[1])
            #
            # dist = xt_norm_view1 + xt_norm_view2 - 2.0 * torch.bmm(xt, xt.transpose(2, 1))
            # triu_indices = torch.triu_indices(dist.shape[1], dist.shape[2], offset=1)
            #
            # return torch.sqrt(dist[:, triu_indices[0], triu_indices[1]]).view(dist.shape[0], -1)

        else:

            delta_x0 = self._x0[node_pairs[0], :] - self._x0[node_pairs[1], :]
            delta_v = self._v[:, node_pairs[0], :] - self._v[:, node_pairs[1], :]

            # temp is a tensor of size len(times_list) x node_pairs.shape[0] x dim
            temp = self.get_xt(times_list=times_list, x0=delta_x0, v=delta_v)
            norm = torch.norm(temp, p=2, dim=2, keepdim=False)

            return norm

    def get_intensity(self, times_list: torch.tensor, node_pairs: torch.tensor):

        # Add an additional axis for beta parameters for time dimension
        intensities = (self._beta[node_pairs[0]] + self._beta[node_pairs[1]]).unsqueeze(0)
        intensities -= self.get_pairwise_distances(times_list=times_list, node_pairs=node_pairs)

        return torch.exp(intensities)

    def get_log_intensity(self, times_list: torch.Tensor, node_pairs: torch.Tensor):

        # Add an additional axis for beta parameters for time dimension
        intensities = -self.get_pairwise_distances(times_list=times_list, node_pairs=node_pairs)
        intensities += (self._beta[node_pairs[0]] + self._beta[node_pairs[1]]).expand(len(times_list), 1)

        return intensities

    def get_intensity_integral(self, x0: torch.Tensor = None, v: torch.Tensor = None, node_pairs: torch.tensor = None):

        if x0 is not None or v is not None:
            raise NotImplementedError("Not implemented for given x0 and v!")

        if node_pairs is None:
            node_pairs = torch.triu_indices(self._nodes_num, self._nodes_num, offset=1)

        delta_x0 = self._x0[node_pairs[0], :] - self._x0[node_pairs[1], :]
        delta_v = self._v[:, node_pairs[0], :] - self._v[:, node_pairs[1], :]
        delta_xt = self.get_xt(times_list=self._bin_boundaries, x0=delta_x0, v=delta_v)

        delta_xt_norm = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
        delta_exp_term = torch.exp(
            self._beta[node_pairs[0]].unsqueeze(0) + self._beta[node_pairs[1]].unsqueeze(0) - delta_xt_norm
        )

        numer = torch.mul(delta_xt_norm, delta_exp_term)
        term1 = torch.divide(numer[1:, :], torch.mul(delta_xt[1:, :, :], delta_v).sum(dim=2))
        term0 = torch.divide(numer[:-1, :], torch.mul(delta_xt[:-1, :, :], delta_v).sum(dim=2))

        # the result is a matrix of size bins_counts x len(node_pairs)
        return torch.sum(term1 - term0)

    def get_negative_log_likelihood(self, time_seq_list: list, node_pairs: torch.tensor):

        integral_term = -self.get_intensity_integral(node_pairs=node_pairs)

        non_integral_term = 0
        for idx in range(node_pairs.shape[1]):
            nodes_pair = node_pairs[:, idx].view(2, 1)
            times_list = time_seq_list[idx]
            non_integral_term += torch.sum(self.get_log_intensity(times_list=times_list, node_pairs=nodes_pair))

        return -(integral_term + non_integral_term)

    def get_model_params(self):

        return {"beta": self._beta, "x0": self._x0, "v": self._v}





