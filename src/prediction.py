from src.learning import LearningModel
from utils import *


class PredictionModel(torch.nn.Module):

    def __init__(self, lm: LearningModel, pred_init_time, pred_last_time, num_of_samples: int = 100):
        super(PredictionModel, self).__init__()

        # Set the learning model
        self._lm = lm
        # Set the parameters of Learning model
        for key, value in self._lm.get_hyperparameters().items():
            setattr(self, key, value)
        # Set the initial and last time points
        self._pred_init_time = pred_init_time
        self._pred_last_time = pred_last_time

        # Sample some time points
        self._num_of_samples = num_of_samples
        self._time_samples = torch.linspace(self._pred_init_time, self._pred_last_time, self._num_of_samples)
        self._time_delta = self._time_samples[1] - self._time_samples[0]

        # A tensor of len(time_samples) x _nodes_num x dim
        self._train_cov_inv = self.get_train_cov_inv()
        self._x_init = self._lm.get_xt(
            events_times_list=torch.as_tensor([self._pred_init_time]*self._nodes_num), x0=self._x0, v=self._v
        )

        self._time_samples_expected_v = self.get_expected_vt(times_list=self._time_samples)

    def get_train_cov_inv(self,  nodes: torch.Tensor = None):

        if nodes is not None:
            raise ValueError("It has been implemented for the whole node set!")

        # Get the bin bounds
        bounds = self._lm.get_bins_bounds()
        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, len(bounds)-1) / 2.

        # B x B matrix
        B_factor = self._lm._get_B_factor(bin_centers1=middle_bounds, bin_centers2=middle_bounds)
        # N x K matrix where K is the community size
        C_factor = self._lm._get_C_factor().T
        # D x D matrix
        D_factor = self._lm._get_D_factor()

        # Some common parameters
        lambda_sq = self._prior_lambda ** 2
        sigma_sq = torch.sigmoid(self._prior_sigma)
        sigma_sq_inv = 1.0 / sigma_sq
        final_dim = self._lm.get_number_of_nodes() * (len(bounds)-1) * self._dim
        reduced_dim = self._prior_C_Q.shape[0] * (len(bounds)-1) * self._dim

        K_factor = torch.kron(B_factor.contiguous(), torch.kron(C_factor.contiguous(), D_factor).contiguous())

        R = torch.eye(reduced_dim) + sigma_sq_inv * K_factor.T @ K_factor
        R_inv = torch.cholesky_inverse(torch.linalg.cholesky(R))

        # Compute the inverse of covariance matrix
        # Computation of the squared Mahalanobis distance: v.T @ inv(D + W @ W.T) @ v
        # It uses Woodbury matrix identity: inv(D + Kf @ Kf.T) = inv(D) - inv(D) @ Kf @ inv(R) @ Kf.T @ inv(D),
        # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
        mahalanobis_term1 = sigma_sq_inv * torch.eye(final_dim)
        mahalanobis_term2 = sigma_sq_inv * K_factor @ R_inv @ K_factor.T * sigma_sq_inv
        train_cov_inv = (1.0 / lambda_sq) * (mahalanobis_term1 - mahalanobis_term2)

        return train_cov_inv

    def get_expected_vt(self, times_list: torch.Tensor):

        nodes = torch.arange(self._nodes_num)

        # Normalize and vectorize the velocities
        v_batch = mean_normalization(torch.index_select(self._v, dim=1, index=nodes))
        v_vect_batch = utils.vectorize(v_batch).flatten()

        # Get the bin bounds
        bounds = self._lm.get_bins_bounds()
        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, len(bounds) - 1) / 2.

        # N x K matrix where K is the community size
        C_factor = self._lm._get_C_factor().T
        # D x D matrix
        D_factor = self._lm._get_D_factor()

        B_test_train_factor = self._lm._get_B_factor(
            bin_centers1=middle_bounds, bin_centers2=times_list.view(1, 1, len(times_list)), only_kernel=True
        )
        test_train_cov = torch.kron(B_test_train_factor, torch.kron(C_factor @ C_factor.T, D_factor @ D_factor.T))

        mean_vt = test_train_cov @ self._train_cov_inv @ v_vect_batch

        return utils.unvectorize(mean_vt, size=(len(times_list), v_batch.shape[1], v_batch.shape[2]))

    def get_expected_displacements(self, times_list: torch.Tensor, nodes: torch.Tensor):

        expected_vt = torch.index_select(self.get_expected_vt(times_list=times_list), dim=1, index=nodes)

        events_bin_indices = torch.div(
            times_list - self._pred_init_time, self._time_delta, rounding_mode='floor'
        ).type(torch.int)
        residual_time = (times_list - self._pred_init_time) % self._time_delta
        events_bin_indices[events_bin_indices == len(self._time_samples) - 1] = len(self._time_samples) - 2

        # Riemann integral for computing average displacement
        xt_disp = torch.cumsum(
            self._time_delta * torch.index_select(self._time_samples_expected_v, dim=1, index=nodes), dim=0
        )
        xt_disp = torch.index_select(xt_disp, dim=0, index=events_bin_indices)

        # Remaining displacement
        remain_disp = torch.mul(expected_vt, residual_time.unsqueeze(1))

        # Get average position
        mean_xt = torch.index_select(self._x_init, dim=0, index=nodes).unsqueeze(0) + xt_disp + remain_disp

        return mean_xt

    def get_log_intensity(self, times_list: torch.Tensor, node_pairs: torch.Tensor):

        # Sum of bias terms
        intensities = torch.index_select(self._beta, dim=0, index=node_pairs[0]) + \
                      torch.index_select(self._beta, dim=0, index=node_pairs[1])

        for idx in range(node_pairs.shape[1]):

            pair_xt = self.get_expected_displacements(
                times_list=times_list[idx].unsqueeze(0), nodes=node_pairs[:, idx]
            ).squeeze(0)
            pair_norm = torch.norm(pair_xt[0, :] - pair_xt[1, :], p=2, dim=0, keepdim=False) ** 2

            intensities[idx] = intensities[idx] - pair_norm

        return intensities

    def get_intensity(self, times_list: torch.Tensor, node_pairs: torch.Tensor):

        return torch.exp(self.get_log_intensity(times_list=times_list, node_pairs=node_pairs))

    def get_intensity_integral(self, nodes: torch.Tensor):

        return self._lm.get_intensity_integral(
            nodes=nodes, x0=self._x_init, v=self._time_samples_expected_v[:-1, :, :],
            beta=self._beta, bin_bounds=self._time_samples
        )

    def get_negative_log_likelihood(self, event_times: torch.Tensor, event_node_pairs: torch.Tensor):

        nodes = torch.arange(self._nodes_num)

        integral_term_all_pairs = -self._lm.get_intensity_integral(
            nodes=nodes, x0=self._x_init, v=self._time_samples_expected_v[:-1, :, :],
            beta=self._beta, bin_bounds=self._time_samples
        )

        integral_term = torch.as_tensor(
            [integral_term_all_pairs[utils.pairIdx2flatIdx(p[0], p[1], self._nodes_num)] for p in event_node_pairs.T],
            dtype=torch.float
        )

        non_integral_term = self.get_log_intensity(times_list=event_times, node_pairs=event_node_pairs)

        return -(non_integral_term + integral_term)



