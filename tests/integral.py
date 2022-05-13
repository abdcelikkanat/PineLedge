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

    # def test_integral_whole(self):
    #
    #     dim = 2
    #     nodes_num = 2
    #     bins_num = 100
    #     # x0 = 0.5*torch.eye(nodes_num) #utils.mean_normalization(torch.rand(size=(nodes_num, dim)))
    #     # v = torch.vstack((-1.0*torch.eye(nodes_num).unsqueeze(0), 1.0*torch.eye(nodes_num).unsqueeze(0))) #utils.mean_normalization(torch.rand(size=(bins_num, nodes_num, dim)))
    #     x0 = torch.randn(size=(nodes_num, dim))
    #     v = torch.randn(size=(bins_num, nodes_num, dim))
    #
    #     x0 = utils.mean_normalization(x0)
    #     v = utils.mean_normalization(v)
    #
    #     beta = torch.zeros(size=(nodes_num, ))
    #
    #     bm = BaseModel(x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=1.0,
    #                    prior_sigma = 2 * torch.rand(size=(1,)) - 1,
    #                    prior_B_sigma = 2 * torch.rand(size=(1,)) - 1,
    #                    prior_C_Q = torch.rand(size=(4, nodes_num)),
    #     )
    #     tl = 0. #+ utils.EPS
    #     tr = 1. #- + utils.EPS
    #
    #     exact = bm.get_intensity_integral(
    #         nodes=torch.as_tensor([0, 1]),
    #     )
    #
    #     print("exact:", exact)
    #
    #     sample_time_list = torch.linspace(tl, tr, steps=10000)
    #     delta_t = sample_time_list[1] - sample_time_list[0]
    #     riemann_integral_sum_lower_bound = 0
    #     for sample_t in sample_time_list[:-1]:
    #         riemann_integral_sum_lower_bound += bm.get_intensity(times_list=torch.as_tensor([sample_t]),
    #                                                              node_pairs=torch.as_tensor([[0, 1]]).T)
    #     riemann_integral_sum_lower_bound = riemann_integral_sum_lower_bound * delta_t
    #
    #     riemann_integral_sum_upper_bound = 0
    #     for sample_t in sample_time_list[1:]:
    #         riemann_integral_sum_upper_bound += bm.get_intensity(times_list=torch.as_tensor([sample_t]),
    #                                                              node_pairs=torch.as_tensor([[0, 1]]).T)
    #     riemann_integral_sum_upper_bound = riemann_integral_sum_upper_bound * delta_t
    #
    #     print("Riemann:", riemann_integral_sum_lower_bound, riemann_integral_sum_upper_bound)

    def test_integral_pair(self):

        dim = 2
        nodes_num = 2
        bins_num = 100
        # x0 = 0.5*torch.eye(nodes_num) #utils.mean_normalization(torch.rand(size=(nodes_num, dim)))
        # v = torch.vstack((-1.0*torch.eye(nodes_num).unsqueeze(0), 1.0*torch.eye(nodes_num).unsqueeze(0))) #utils.mean_normalization(torch.rand(size=(bins_num, nodes_num, dim)))
        x0 = torch.randn(size=(nodes_num, dim))
        v = torch.randn(size=(bins_num, nodes_num, dim))

        x0 = utils.mean_normalization(x0)
        v = utils.mean_normalization(v)

        beta = torch.zeros(size=(nodes_num, ))

        bm = BaseModel(x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=1.0,
                       prior_sigma = 2 * torch.rand(size=(1,)) - 1,
                       prior_B_sigma = 2 * torch.rand(size=(1,)) - 1,
                       prior_C_Q = torch.rand(size=(4, nodes_num)),
        )
        tl = 0.67 #+ utils.EPS
        tr = 0.98 #- + utils.EPS
        # exact = bm.get_intensity_integral_for(i=0, j=1, interval=torch.as_tensor([tl, tr]))
        # exact = bm.get_intensity_integral(
        #     nodes=torch.as_tensor([0, 1]),
        # )
        # print("Bin integrals: ", ss)
        exact = bm.get_intensity_integral_for(i=0, j=1, interval=torch.as_tensor([tl, tr]))
        # print(ss, ff)
        print("exact:", exact)
        sample_time_list = torch.linspace(tl, tr, steps=10000)
        delta_t = sample_time_list[1] - sample_time_list[0]
        riemann_integral_sum_lower_bound = 0
        for sample_t in sample_time_list[:-1]:
            riemann_integral_sum_lower_bound += bm.get_intensity(times_list=torch.as_tensor([sample_t]),
                                                                 node_pairs=torch.as_tensor([[0, 1]]).T)
        riemann_integral_sum_lower_bound = riemann_integral_sum_lower_bound * delta_t

        riemann_integral_sum_upper_bound = 0
        for sample_t in sample_time_list[1:]:
            riemann_integral_sum_upper_bound += bm.get_intensity(times_list=torch.as_tensor([sample_t]),
                                                                 node_pairs=torch.as_tensor([[0, 1]]).T)
        riemann_integral_sum_upper_bound = riemann_integral_sum_upper_bound * delta_t

        print("riemann:", riemann_integral_sum_lower_bound, riemann_integral_sum_upper_bound)


if __name__ == '__main__':
    unittest.main()
