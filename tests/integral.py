import unittest
import torch
import numpy as np
import random
import functools
import utils
from src.base import BaseModel


class AlgebraicEquivalence(unittest.TestCase):

    def __init__(self, methodName, seed=123):

        super(AlgebraicEquivalence, self).__init__(methodName=methodName)

        self.assertEqualTensor = functools.partial(torch.testing.assert_close, rtol=0.01, atol=0.01)

        self._seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)

    def test_zero_mat(self):

        self.assertEqualTensor(torch.zeros(size=(2, 2)), torch.zeros(size=(2, 2)))

    def test_integral_whole(self):

        dim = 2
        nodes_num = 2
        bins_num = 100
        K = 4
        tl = 0.
        tu = 1.

        x0 = torch.randn(size=(nodes_num, dim))
        v = torch.randn(size=(bins_num, nodes_num, dim))

        x0 = utils.mean_normalization(x0)
        v = utils.mean_normalization(v)
        beta = torch.zeros(size=(nodes_num, ))

        bm = BaseModel(x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=tu)

        exact = bm.get_intensity_integral(
            nodes=torch.as_tensor([0, 1]),
        )

        sample_time_list = torch.linspace(tl, tu, steps=10000)
        delta_t = sample_time_list[1] - sample_time_list[0]
        riemann_integral_sum_lower_bound = 0
        for sample_t in sample_time_list[:-1]:
            riemann_integral_sum_lower_bound += bm.get_intensity(times_list=torch.as_tensor([sample_t]),
                                                                 node_pairs=torch.as_tensor([[0, 1]]).T)
        riemann_integral_sum_lower_bound = riemann_integral_sum_lower_bound * delta_t

        self.assertEqualTensor(exact.data, riemann_integral_sum_lower_bound.data)

    def test_integral_pair(self):

        dim = 2
        nodes_num = 2
        bins_num = 100
        tl = 0.22
        tu = 0.98

        x0 = torch.randn(size=(nodes_num, dim))
        v = torch.randn(size=(bins_num, nodes_num, dim))

        x0 = utils.mean_normalization(x0)
        v = utils.mean_normalization(v)

        beta = torch.zeros(size=(nodes_num, ))

        bm = BaseModel(x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=1.0)

        exact = bm.get_intensity_integral_for(i=0, j=1, interval=torch.as_tensor([tl, tu]))

        sample_time_list = torch.linspace(tl, tu, steps=10000)
        delta_t = sample_time_list[1] - sample_time_list[0]
        riemann_integral_sum_lower_bound = 0
        for sample_t in sample_time_list[:-1]:
            riemann_integral_sum_lower_bound += bm.get_intensity(times_list=torch.as_tensor([sample_t]),
                                                                 node_pairs=torch.as_tensor([[0, 1]]).T)
        riemann_integral_sum_lower_bound = riemann_integral_sum_lower_bound * delta_t

        self.assertEqualTensor(exact.data, riemann_integral_sum_lower_bound.data[0])

    def test_integral_whole2(self):

        dim = 2
        nodes_num = 2
        bins_num = 100
        K = 4
        tl = 0.
        tu = 1.

        x0 = torch.randn(size=(nodes_num, dim))
        v = 0 * torch.randn(size=(bins_num, nodes_num, dim))
        beta = 1 * torch.randn(size=(nodes_num,))

        x0 = utils.mean_normalization(x0)
        v = utils.mean_normalization(v)

        bm = BaseModel(x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=tu)

        exact = bm.get_intensity_integral(
            nodes=torch.as_tensor([0, 1]),
        )

        expected = torch.as_tensor([ torch.exp( beta[0] + beta[1] - torch.sum((x0[0, :] - x0[1, :])**2) ) ])
        print(exact.data, expected.data)
        self.assertEqualTensor(exact.data, expected.data)


if __name__ == '__main__':
    unittest.main()
