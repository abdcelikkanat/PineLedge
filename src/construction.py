import torch
import numpy as np
from src.nhpp import NHPP
from src.base import BaseModel
import pickle as pkl
from utils import *


class InitialPositionVelocitySampler:

    def __init__(self, dim: int, bins_num: int, cluster_sizes: list,
                 prior_lambda: float, prior_sigma: float, prior_B_sigma: float,
                 device: torch.device = "cpu", verbose: bool = False, seed: int = 0):

        self.__dim = dim
        self.__bins_num = bins_num
        self.__cluster_sizes = cluster_sizes
        self.__prior_lambda = prior_lambda
        self.__prior_sigma = prior_sigma
        self.__prior_B_sigma = prior_B_sigma
        self.__time_interval_lengths = [1]*bins_num

        self.__nodes_num = sum(cluster_sizes)
        self.__K = len(cluster_sizes)
        self.__x0 = None
        self.__v = None

        self.__device = device
        self.__verbose = verbose
        self.__seed = seed

        self.__sample()

    def __sample(self):

        # Get the factor of B matrix, (bins)
        bin_centers = torch.arange(0.5, 0.5*(self.__bins_num+1), 0.5)
        # Add a center point for the initial position
        bin_centers = torch.hstack((torch.as_tensor([-0.5]), bin_centers))
        bin_centers = bin_centers.view(1, 1, self.__bins_num+1)

        B_factor = BaseModel.get_B_factor(
            bin_centers1=bin_centers, bin_centers2=bin_centers,
            prior_B_sigma=torch.as_tensor(self.__prior_B_sigma), only_kernel=True
        )

        # Get the factor of C matrix, (nodes)
        prior_C_Q = torch.ones(size=(self.__nodes_num, self.__K), dtype=torch.float) * (-1e6)
        for k in range(self.__K):
            prior_C_Q[range(sum(self.__cluster_sizes[:k]), sum(self.__cluster_sizes[:k + 1])), k] = 1e6
        C_factor = BaseModel.get_C_factor(prior_C_Q)

        # Get the factor of D matrix, (dimension)
        D_factor = BaseModel.get_D_factor(dim=self.__dim)

        # Sample the initial position and velocity vectors
        final_dim = self.__nodes_num * (self.__bins_num+1) * self.__dim
        cov_factor = (self.__prior_lambda) * torch.kron(torch.kron(B_factor, C_factor), D_factor)
        cov_diag = (self.__prior_lambda ** 2) * (self.__prior_sigma ** 2) * torch.ones(final_dim)
        lmn = torch.distributions.LowRankMultivariateNormal(
            loc=torch.zeros(size=(final_dim,)),
            cov_factor=cov_factor,
            cov_diag=cov_diag
        )

        sample = lmn.sample().reshape(shape=(self.__bins_num + 1, self.__nodes_num, self.__dim))
        self.__x0, self.__v = torch.split(sample, [1, self.__bins_num])
        self.__x0 = self.__x0.squeeze(0)

    def get_x0(self):

        return self.__x0

    def get_v(self):

        return self.__v

    def get_last_time(self):

        return self.__bins_num


class ConstructionModel(BaseModel):

    def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, last_time: float,
                 bins_num: int, device: torch.device = "cpu", verbose: bool = False, seed: int = 0):

        super(ConstructionModel, self).__init__(
            x0=x0,
            v=v,
            beta=beta,
            last_time=last_time,
            bins_num=bins_num,
            device=device,
            verbose=verbose,
            seed=seed
        )

        self.__events = self.__sample_events()

    def __get_critical_points(self, i: int, j: int, x: torch.tensor):

        bin_bounds = self.get_bins_bounds()

        # Add the initial time point
        critical_points = []

        for idx in range(self._bins_num):

            interval_init_time = bin_bounds[idx]
            interval_last_time = bin_bounds[idx+1]

            # Add the initial time point of the interval
            critical_points.append(interval_init_time)

            # Get the differences
            delta_idx_x = x[idx, i, :] - x[idx, j, :]
            delta_idx_v = self._v[idx, i, :] - self._v[idx, j, :]

            # For the model containing only position and velocity
            # Find the point in which the derivative equal to 0
            t = - np.dot(delta_idx_x, delta_idx_v) / (np.dot(delta_idx_v, delta_idx_v) + utils.EPS) + interval_init_time

            if interval_init_time < t < interval_last_time:
                critical_points.append(t)

        # Add the last time point
        critical_points.append(bin_bounds[-1])

        return critical_points

    def __sample_events(self, nodes: torch.tensor = None) -> dict:

        if nodes is not None:
            raise NotImplementedError("It must be implemented for given specific nodes!")

        node_pairs = torch.triu_indices(row=self._nodes_num, col=self._nodes_num, offset=1, device=self._device)

        # Upper triangular matrix of lists
        events_time = {i: {j: [] for j in range(i+1, self._nodes_num)} for i in range(self._nodes_num-1)}
        # Get the positions at the beginning of each time bin for every node
        x = self.get_xt(
            events_times_list=self.get_bins_bounds()[:-1].repeat(self._nodes_num, ),
            x0=torch.repeat_interleave(self._x0, repeats=self._bins_num, dim=0),
            v=torch.repeat_interleave(self._v, repeats=self._bins_num, dim=1)
        ).reshape((self._nodes_num, self._bins_num,  self._dim)).transpose(0, 1)

        for i, j in zip(node_pairs[0], node_pairs[1]):
            # Define the intensity function for each node pair (i,j)
            intensity_func = lambda t: self.get_intensity(
                times_list=torch.as_tensor([t]), node_pairs=torch.as_tensor([[i], [j]])
            ).item()
            # Get the critical points
            critical_points = self.__get_critical_points(i=i, j=j, x=x)
            # Simulate the src
            nhpp_ij = NHPP(
                intensity_func=intensity_func, critical_points=critical_points, seed=self._seed+i*self._nodes_num + j
            )
            ij_events_time = nhpp_ij.simulate()
            # Add the event times
            events_time[i.item()][j.item()].extend(ij_events_time)

        return events_time

    def get_events(self):

        return self.__events

    def save(self, folder_path):
        events, pairs = [], []
        for i, j in utils.pair_iter(n=self._nodes_num):
            pair_events = self.__events[i][j]
            if len(pair_events):
                pairs.append([i, j])
                events.append(pair_events)

        with open(os.path.join(folder_path, "pairs.pkl"), 'wb') as f:
            pkl.dump(pairs, f)

        with open(os.path.join(folder_path, "events.pkl"), 'wb') as f:
            pkl.dump(events, f)