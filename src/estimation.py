import torch
# from src.base import BaseModel
# from src.construction import ConstructionModel
from src.learning import LearningModel
# from datasets.datasets import Dataset
# from src.animation import Animation
# from torch.utils.data import DataLoader
# from src.experiments import Experiments
import utils.utils
import numpy as np
import pandas as pd
import os
import sys
import utils


class Estimation(torch.nn.Module):
    def __init__(self, lm: LearningModel, init_time, split_time, last_time, sample_num=100):
        super(Estimation, self).__init__()

        self._lm = lm
        self._init_time = init_time
        self._split_time = split_time
        self._last_time = last_time

        # Get the number of nodes
        self._nodes_num = self._lm.get_number_of_nodes()
        # Get the initial position
        self._x0 = self._lm.get_x0()
        # Get the velocity vectors
        self._v = self._lm.get_v()
        # Get the number of bin size for the training set
        self._bin_num = self._lm.get_num_of_bins()
        # Get the bin bounds for the training set
        self._bounds = self._lm.get_bins_bounds()
        # Get the dimension size
        self._dim = self._lm.get_dim()

        # Get the position at the splitting time point
        self._x_split_time = self._lm.get_xt(times_list=torch.as_tensor([self._split_time])).squeeze(0)

        # Sample time points for Reimann integral
        self._time_samples = torch.linspace(self._init_time, self._last_time, sample_num)
        self._time_delta = self._time_samples[1] - self._time_samples[0]
        self._mean_v_time_samples = self._get_mean_vt(times_list=self._time_samples)

    def get_log_intensity(self, time_list: list, node_pairs: list):

        node_pairs = torch.as_tensor(node_pairs)
        time_list = torch.as_tensor(time_list, dtype=torch.float)
        train_time_idx = time_list <= self._split_time
        test_time_idx = time_list > self._split_time
        train_time_list = time_list[train_time_idx]
        test_time_list = time_list[test_time_idx]

        intensity = torch.zeros(size=(len(time_list), node_pairs.shape[1]), )
        if len(train_time_list) > 0:
            intensity[train_time_idx, :] = self._lm.get_log_intensity(times_list=time_list, node_pairs=node_pairs)

        if len(test_time_list) > 0:
            xt = self.get_xt(time_list=time_list)
            xt_delta = xt[:, node_pairs[0], :] - xt[:, node_pairs[1], :]
            norm = torch.norm(xt_delta, p=2, dim=2, keepdim=False) ** 2

            intensity[test_time_idx, :] = (self._lm._beta[node_pairs[0]] + self._lm._beta[node_pairs[1]]).expand(len(time_list), node_pairs.shape[1])
            intensity[test_time_idx, :] -= norm

        return intensity

    def _get_mean_vt(self, times_list: torch.Tensor, nodes: torch.Tensor = None):

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (self._bounds[1:] + self._bounds[:-1]).view(1, 1, self._bin_num) / 2.

        # A: T x T matrix
        _, inv_A = self._lm._LearningModel__get_A(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=True
        )

        # B: D x D matrix
        inv_B = self._lm._LearningModel__get_B()

        # C: len(nodes) x len(nodes) matrix
        inv_C = self._lm._LearningModel__get_C(nodes=nodes)

        # # Compute the inverse of kernel/covariance matrix
        kernel_train_inv = torch.kron(inv_A.contiguous(), torch.kron(inv_B.contiguous(), inv_C.contiguous()))

        # Compute the inverse of kernel/covariance matrix
        A_test_train = self._lm._LearningModel__get_A(
            bin_centers1=middle_bounds, bin_centers2=times_list.view(1, 1, len(times_list)), get_inv=False
        )

        kernel_test_train = torch.kron(
            A_test_train.contiguous(), torch.kron(
                torch.linalg.inv(inv_B).contiguous(), torch.linalg.inv(inv_C).contiguous()
            )
        )

        if nodes is None:
            batch_v = utils.mean_normalization(self._v)
        else:
            batch_v = utils.mean_normalization(self._v)[:, nodes, :]
        v_vect = utils.vectorize(batch_v).flatten()

        # A tensor of size len(times_list) x len(v_vect)
        mean_vt = kernel_test_train @ kernel_train_inv @ v_vect

        return utils.unvectorize(mean_vt, size=(len(times_list), batch_v.shape[1], batch_v.shape[2]))

    def _get_displacement(self, times_list: torch.Tensor, nodes: torch.Tensor = None):

        mean_vt = self._get_mean_vt(times_list=times_list, nodes=nodes)

        # Find the indices of given time points
        times_list_indices = torch.bucketize(times_list, boundaries=self._time_samples[1:-1], right=True)
        # Compute the distance of the time points to the initial time of the intervals that they lay on
        time_remainders = times_list - self._time_samples[times_list_indices]

        # Riemann integral for computing average displacement
        #xt_disp = torch.sum(self._time_delta * mean_vt, dim=0, keepdim=False)
        xt_disp = torch.cumsum(self._time_delta * self._mean_v_time_samples, dim=0)[times_list_indices, :, :].squeeze(0)

        # Remaining displacement
        remaining_displacement = mean_vt * time_remainders[:, None, None]

        # Get average position
        mean_xt = xt_disp + remaining_displacement

        return mean_xt

    def get_xt(self, time_list: torch.Tensor):

        train_idx = time_list <= self._split_time
        test_idx = time_list > self._split_time
        train_time_list = time_list[train_idx]
        test_time_list = time_list[test_idx]

        xt = torch.zeros(size=(len(time_list), self._nodes_num, self._dim))
        xt[train_idx, :, :] = self._lm.get_xt(times_list=train_time_list)
        displacement = self._get_displacement(times_list=test_time_list, nodes=torch.arange(self._nodes_num))
        xt[test_idx, :, :] = self._x_split_time + displacement

        return xt


class PredictionModel(torch.nn.Module):

    def __init__(self, lm: LearningModel, test_init_time, test_last_time, num_of_samples=100):
        super(PredictionModel, self).__init__()

        self._lm = lm
        params = self._lm.get_hyperparameters()

        self._x0 = params['_x0']
        self._v = params['_v']
        self._beta = params['_beta']
        self._bins_rwidth = params['_bins_rwidth']
        self._train_init_time = params['_init_time']
        self._train_last_time = params['_last_time']

        self._prior_rbf_sigma = params.get('prior_rbf_sigma', None)
        self._prior_periodic_sigma = params.get('_prior_periodic_sigma', None)
        self._prior_B_L = params['_prior_B_L']
        self._prior_C_Q = params['_prior_C_Q']
        # self._prior_C_lambda = params['_prior_C_lambda']

        self._nodes_num = params['_nodes_num']
        self._dim = params['_dim']
        self._seed = params['_seed']

        # Get the number of bin size
        self.__bin_num = lm.get_num_of_bins()
        # Get the bin bounds
        self.__bounds = lm.get_bins_bounds()

        self._test_init_time = test_init_time
        self._test_last_time = test_last_time
        self._num_of_samples = num_of_samples

        self._time_samples = torch.linspace(self._test_init_time, self._test_last_time, self._num_of_samples)
        self._time_delta = self._time_samples[1] - self._time_samples[0]

        # Get the number of bin size for the training set
        self._bin_num = self._lm.get_num_of_bins()

        # Get the bin bounds for the training set
        self._bounds = self._lm.get_bins_bounds()

        # Compute the initial position
        self._x_init = self._lm.get_xt(times_list=torch.as_tensor([self._test_init_time])).squeeze(0)
        # x1 = self._x_init[0, :]
        # x2 = self._x_init[1, :]
        # print("init", self._test_init_time, torch.dot(x1-x2, x1-x2), x1, x2)
        # A tensor of len(time_samples) x _nodes_num x dim
        self._mean_v_samples = self.get_mean_vt(times_list=self._time_samples, nodes=torch.arange(self._nodes_num))

    def get_pred_beta(self, times_list: torch.Tensor):

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (self._bounds[1:] + self._bounds[:-1]).view(1, self._bin_num) / 2.

        # A: T x T matrix
        K, inv_K = self._lm.get_beta_kernel(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=True
        )

        # Compute the inverse of kernel/covariance matrix
        K_test_train = self._lm.get_beta_kernel(
            bin_centers1=middle_bounds, bin_centers2=times_list.view(1, len(times_list)), get_inv=False
        )

        beta = K_test_train @ inv_K @ self._beta

        return beta

    def get_negative_log_likelihood(self, times_list: torch.Tensor, node_pair: torch.Tensor):

        assert len(node_pair) == 2, "The input must be a node pair!"

        # pred_beta = self.get_pred_beta(times_list=times_list)
        # print(pred_beta)
        # print("Print beta: ", times_list.shape, pred_beta.shape)

        integral_term = -self._lm.get_intensity_integral(
            x0=self._x_init, v=self._mean_v_samples[:-1, :, :], beta=self._beta, node_pairs=node_pair,
            bin_bounds=self._time_samples #torch.as_tensor([self._test_init_time, self._test_last_time])
        ).sum()

        # print("s", integral_term.shape)

        nll = self.get_log_intensity(times_list=times_list, node_pair=node_pair, beta=self._beta)

        nll += integral_term

        return nll

    def get_log_intensity(self, times_list: torch.Tensor, node_pair, beta):

        assert len(node_pair) == 2, "The input must be 2-nodes."

        xt = self.get_mean_displacement(times_list=times_list, nodes=node_pair.view(2,))
        # xt = self._lm.get_xt(times_list=torch.as_tensor([self._test_init_time])).squeeze(0)[node_pair.t(), :]
        # xt = torch.zeros(size=xt.shape)
        delta_xt = xt[:, 0, :] - xt[:, 1, :]
        norm = torch.norm(delta_xt, p=2, dim=1, keepdim=False) ** 2
        # Add an additional axis for beta parameters for time dimension
        intensities = (beta[node_pair[0]] + beta[node_pair[1]]).expand(len(times_list), ) - norm
        # print("shape of: ", beta[:, node_pair[0]].shape, norm.shape)
        # intensities = beta[:, node_pair[0]] + beta[:, node_pair[1]] - norm
        # intensities = -norm
        return intensities

    def get_mean_displacement(self, times_list: torch.Tensor, nodes: torch.Tensor):

        mean_vt = self.get_mean_vt(times_list=times_list, nodes=nodes)

        times_list_indices = torch.bucketize(times_list, boundaries=self._time_samples[1:-1], right=True)
        # Compute the distance of the time points to the initial time of the intervals that they lay on
        time_remainders = times_list - self._time_samples[times_list_indices]

        # Riemann integral for computing average displacement
        #xt_disp = torch.sum(self._time_delta * mean_vt, dim=0, keepdim=False)
        xt_disp = torch.cumsum(self._time_delta * self._mean_v_samples[:, nodes, :], dim=0)[times_list_indices, :, :].squeeze(0)

        # Remaining displacement
        remain_disp = mean_vt * time_remainders[:, None, None]

        # Get average position
        mean_xt = self._x_init[nodes, :].unsqueeze(0) + xt_disp + remain_disp

        return mean_xt

    def get_mean_vt(self, times_list: torch.Tensor, nodes:torch.Tensor = None):

        if nodes is None:
            nodes = torch.arange(self._nodes_num)

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (self._bounds[1:] + self._bounds[:-1]).view(1, 1, self._bin_num) / 2.

        # A: T x T matrix
        A, inv_A = self._lm._LearningModel__get_A(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds, cholesky=True
        )

        # B: D x D matrix -> self._prior_B_U: D x D upper triangular matrix
        # B = self._prior_B_U @ self._prior_B_U.t()
        B_L = torch.tril(self._prior_B_L, diagonal=-1) + torch.diag(torch.diag(self._prior_B_L) ** 2)
        inv_B = torch.cholesky_inverse(B_L)

        # C: N x N matrix
        C_Q = self._prior_C_Q[nodes, :]
        # C_D = torch.diag(self._prior_C_lambda)
        C_inv_D = torch.eye(n=len(nodes))  # torch.diag(1.0 / self.__prior_C_lambda.expand(len(nodes)))
        # C = C_D + (C_Q @ C_Q.t())
        # By Woodbury matrix identity
        invDQ = C_inv_D @ C_Q
        inv_C = C_inv_D - invDQ @ torch.inverse(torch.eye(C_Q.shape[1]) + C_Q.t() @ C_inv_D @ C_Q) @ invDQ.t()

        # # Compute the inverse of kernel/covariance matrix
        kernel_train_inv = torch.kron(inv_A.contiguous(), torch.kron(inv_B.contiguous(), inv_C.contiguous()))

        # Compute the inverse of kernel/covariance matrix
        A_test_train = self._lm._LearningModel__get_A(
            bin_centers1=middle_bounds, bin_centers2=times_list.view(1, 1, len(times_list)), get_inv=False
        )

        kernel_test_train = torch.kron(
            A_test_train.contiguous(), torch.kron(
                torch.linalg.inv(inv_B).contiguous(), torch.linalg.inv(inv_C).contiguous()
            )
        )

        # kernel_test = 1.0 * torch.kron(torch.linalg.inv(inv_B).contiguous(), torch.linalg.inv(inv_C).contiguous())

        batch_v = self._v[:, nodes, :]
        v_vect = utils.vectorize(batch_v).flatten()

        # A tensor of size len(times_list) x len(v_vect)
        mean_vt = kernel_test_train @ kernel_train_inv @ v_vect
        # print("xx: ", self._v.shape, mean_vt.shape)

        return utils.unvectorize(mean_vt, size=(len(times_list), batch_v.shape[1], batch_v.shape[2]))

    def get_positions(self, time_list):

        train_idx = time_list <= self._train_last_time
        test_idx = time_list > self._train_last_time
        train_time_list = time_list[train_idx]
        test_time_list = time_list[test_idx]

        xt = torch.zeros(size=(len(time_list), self._nodes_num, self._dim))
        xt[train_idx, :, :] = self._lm.get_xt(times_list=train_time_list)
        xt[test_idx, :, :] = self.get_mean_displacement(times_list=test_time_list, nodes=torch.arange(self._nodes_num))

        return xt