import unittest
import torch
import numpy as np
import random
import functools
from src.base import BaseModel
import utils


class BaseTest(unittest.TestCase):

    def __init__(self, methodName, seed=123):

        super(BaseTest, self).__init__(methodName=methodName)

        self.assertEqualTensor = functools.partial(torch.testing.assert_close, rtol=0.01, atol=0.01)

        # Set the seed value to control the randomness
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)

        # Define the base model
        dim = 2
        nodes_num = 2
        bins_num = 2
        last_time = 10.

        x0 = torch.randn(size=(nodes_num, dim))
        v = torch.randn(size=(bins_num, nodes_num, dim))
        beta = torch.randn(size=(nodes_num,))

        self._bm = BaseModel(
            x0=utils.mean_normalization(x0), v=utils.mean_normalization(v),
            beta=beta, bins_num=bins_num, last_time=last_time
        )

    def test_xt(self):

        events_times_list = torch.as_tensor([1., 6.])
        x0 = torch.as_tensor([[0, 1], [0, -1]])
        v = torch.as_tensor([[[0, 1], [0, -1]], [[0, 1], [0, -1]]])
        pred_xt = self._bm.get_xt(events_times_list=events_times_list, x0=x0, v=v)

        true_xt = torch.as_tensor([[0., 2], [0, -7]])
        self.assertEqualTensor(true_xt, pred_xt)

    def test_xt(self):

        events_times_list = torch.as_tensor([1., 6., 8.])
        x0 = torch.as_tensor([[0, 1], [0, -1], [3, -5]])
        v = torch.as_tensor([[[0, 1], [0, -1], [2, 1]], [[0, 1], [0, -1], [6, -4]]])
        pred_xt = self._bm.get_xt(events_times_list=events_times_list, x0=x0, v=v)
        print(pred_xt)
        true_xt = torch.as_tensor([[0., 2], [0, -7], [31, -12]])
        self.assertEqualTensor(true_xt, pred_xt)

    def test_log_intensity_sum(self):

        events = [[[1., 2.2], [6.1]], [[2.3], [7.1]],]
        node_pairs = torch.as_tensor([[0], [1]])
        events_count = torch.as_tensor(
            [[len(bin_events) for bin_events in pair_events] for pair_events in events]
        )

        alpha1 = torch.as_tensor([
            [torch.sum(utils.remainder(x=torch.as_tensor(bin_events), y=self._bm._bin_width)) if len(bin_events) else 0 for bin_events in pair_events] for pair_events in events
        ])
        alpha2 = torch.as_tensor([
            [torch.sum(utils.remainder(x=torch.as_tensor(bin_events), y=self._bm._bin_width) ** 2) if len(bin_events) else 0 for bin_events in pair_events] for pair_events in events
        ])

        scalable_version = self._bm.get_log_intensity_sum(
            node_pairs=node_pairs, events_count=events_count, alpha1=alpha1, alpha2=alpha2
        )

        times_list = torch.as_tensor([e for pair_events in events for bin_events in pair_events for e in bin_events])
        unscalable_version = self._bm.get_log_intensity(
            times_list=times_list, node_pairs=node_pairs.repeat(1, len(times_list))
        ).sum()

        print(unscalable_version, scalable_version)

    # def test_xt(self):
    #
    #     computed_xt = self.get_xt(
    #         times_list=torch.as_tensor([0, 1, 2, 3, 4]),
    #         x0=torch.as_tensor([[-1, 0], [1, 0]]),
    #         v=torch.as_tensor([[[2.0, 0], [-2.0, 0]], [[-3.0, 0], [3.0, 0]]]),
    #         bin_bounds=torch.as_tensor([0, 2, 4])
    #     )
    #
    #     correct_xt = torch.as_tensor(
    #         [
    #             [
    #                 [-1, 0], [1, 0]
    #             ],
    #             [
    #                 [1.0, 0], [-1.0, 0]
    #             ],
    #             [
    #                 [3.0, 0], [-3.0, 0]
    #             ],
    #             [
    #                 [0, 0], [0, 0]
    #             ],
    #             [
    #                 [-3, 0], [3, 0]
    #             ]
    #         ]
    #     )
    #
    #     self.assertEqualTensor(correct_xt, computed_xt)

    # def test_xt(self):
    #
    #     self._bins_num = 2
    #     self._bin_width = 0.5
    #     computed_xt = self.get_xt(
    #         events_times_list=torch.as_tensor([0, 1,]),
    #         x0=torch.as_tensor([[-1, 0], [1, 0]]),
    #         v=torch.as_tensor([[2.0, 0], [-2.0, 0], ]),
    #     )
    #
    #     correct_xt = torch.as_tensor(
    #         [
    #             [
    #                 [-1, 0], [1, 0]
    #             ],
    #             [
    #                 [1.0, 0], [-1.0, 0]
    #             ],
    #             [
    #                 [3.0, 0], [-3.0, 0]
    #             ],
    #             [
    #                 [0, 0], [0, 0]
    #             ],
    #             [
    #                 [-3, 0], [3, 0]
    #             ]
    #         ]
    #     )
    #
    #     self.assertEqualTensor(correct_xt, computed_xt)


if __name__ == '__main__':
    unittest.main()
