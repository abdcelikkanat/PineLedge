import torch
import numpy as np
from src.nhpp import NHPP
from src.base import BaseModel
from utils.constants import const
import pickle as pkl


class ConstructionModel(BaseModel):

    def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, last_time: float, bins_rwidth: int or torch.Tensor, seed: int = 0):

        super(ConstructionModel, self).__init__(
            x0=x0,
            v=v,
            beta=beta,
            bins_rwidth=bins_rwidth,
            last_time=last_time,
            seed=seed
        )

        self.__events = self.__sample_events()

    def __get_critical_points(self, i: int, j: int, x: torch.tensor):

        bin_bounds = self.get_bins_bounds()

        # Add the initial time point
        critical_points = []

        for idx in range(self.get_num_of_bins()):

            interval_init_time = bin_bounds[idx]
            interval_last_time = bin_bounds[idx+1]

            # Add the initial time point of the interval
            critical_points.append(interval_init_time)

            # Get the differences
            delta_idx_x = x[idx, i, :] - x[idx, j, :]
            delta_idx_v = self._v[idx, i, :] - self._v[idx, j, :]

            # For the model containing only position and velocity
            # Find the point in which the derivative equal to 0
            t = - np.dot(delta_idx_x, delta_idx_v) / (np.dot(delta_idx_v, delta_idx_v) + const.eps) + interval_init_time

            if interval_init_time < t < interval_last_time:
                critical_points.append(t)

        # Add the last time point
        critical_points.append(bin_bounds[-1])

        return critical_points

    def __sample_events(self, nodes: torch.tensor = None) -> dict:

        if nodes is not None:
            raise NotImplementedError("It must be implemented for given specific nodes!")

        nodes = torch.arange(self._nodes_num)
        node_pairs = torch.triu_indices(row=self._nodes_num, col=self._nodes_num, offset=1)

        # Upper triangular matrix of lists
        events_time = {i.item(): {j.item(): [] for j in node_pairs[1]} for i in node_pairs[0]}
        # Get the positions at the beginning of each time bin for every node
        x = self.get_xt(times_list=self.get_bins_bounds()[:-1])

        for i, j in zip(node_pairs[0], node_pairs[1]):
            # Define the intensity function for each node pair (i,j)
            intensity_func = lambda t: self.get_intensity(torch.as_tensor([t]), torch.as_tensor([[i], [j]])).item()
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

    def save(self, file_path):

        with open(file_path, 'wb') as f:
            pkl.dump(
                {"pairs": torch.triu_indices(row=self._nodes_num, col=self._nodes_num, offset=1),
                 "events": self.__events}, f
            )