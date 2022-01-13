import unittest
import torch
import numpy as np
import random
import functools
from src.base import BaseModel


class BaseTest(unittest.TestCase, BaseModel):

    def __init__(self, methodName, seed=123):

        super(BaseTest, self).__init__(methodName=methodName)

        self.assertEqualTensor = functools.partial(torch.testing.assert_close, rtol=0.01, atol=0.01)

        self._seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)

    def test_xt(self):

        computed_xt = self.get_xt(
            times_list=torch.as_tensor([0, 1, 2, 3, 4]),
            x0=torch.as_tensor([[-1, 0], [1, 0]]),
            v=torch.as_tensor([[[2.0, 0], [-2.0, 0]], [[-3.0, 0], [3.0, 0]]]),
            bin_bounds=torch.as_tensor([0, 2, 4])
        )

        correct_xt = torch.as_tensor(
            [
                [
                    [-1, 0], [1, 0]
                ],
                [
                    [1.0, 0], [-1.0, 0]
                ],
                [
                    [3.0, 0], [-3.0, 0]
                ],
                [
                    [0, 0], [0, 0]
                ],
                [
                    [-3, 0], [3, 0]
                ]
            ]
        )

        self.assertEqualTensor(correct_xt, computed_xt)


if __name__ == '__main__':
    unittest.main()
