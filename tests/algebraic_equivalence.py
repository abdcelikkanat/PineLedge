import unittest
import torch
import numpy as np
import random
import functools


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

    def test_mixed_kron_matrix_vector_prod(self):
        A = torch.rand(size=(5, 3))
        B = torch.rand(size=(4, 2))
        V = torch.rand(size=(3, 2))

        # print( torch.kron(B.transpose(0, 1).contiguous(), A), C.flatten() )
        true_vect = torch.matmul(torch.kron(B, A), V.transpose(0, 1).flatten())
        pred_vect = torch.matmul(torch.matmul(A, V), B.transpose(0, 1)).transpose(0, 1).flatten()

        self.assertEqualTensor(true_vect, pred_vect)

    def test_mixed_kron_triple_matrix_vector_prod_one_side(self):
        '''

        '''

        def __vect(x):

            return x.transpose(-2, -1).flatten(-2)

        A = torch.rand(size=(12, 9))
        B = torch.rand(size=(3, 3))
        C = torch.rand(size=(3, 4))
        V = torch.rand(size=(9, 4, 3))

        v = __vect(V).flatten()
        true_vect = torch.matmul(torch.kron(A, torch.kron(B, C)), v)

        pred_vect = __vect(torch.matmul(
                __vect(torch.matmul(torch.matmul(C.unsqueeze(0), V), B.transpose(0, 1).unsqueeze(0))).transpose(0, 1),
                A.transpose(0, 1)
            ))

        self.assertEqualTensor(true_vect, pred_vect)

    def test_mixed_kron_triple_matrix_vector_prod(self):
        '''
        Claim: vect(V)^t ( A \kron B \kron C ) vect(V) is equal to vec(V)^t vec( (CVB^t)A^t )

        '''

        def __vect(x):

            return x.transpose(-2, -1).flatten(-2)

        A = torch.rand(size=(12, 9))
        B = torch.rand(size=(3, 3))
        C = torch.rand(size=(3, 4))
        V = torch.rand(size=(9, 4, 3))

        v = __vect(V).flatten()
        true_vect = torch.matmul(torch.matmul(v, torch.kron(A, torch.kron(B, C))), v)

        pred_vect = torch.matmul(
            v,
            __vect(torch.matmul(
                __vect(torch.matmul(torch.matmul(C.unsqueeze(0), V), B.transpose(0, 1).unsqueeze(0))).transpose(0, 1),
                A.transpose(0, 1)
            ))
        )

        self.assertEqualTensor(true_vect, pred_vect)


if __name__ == '__main__':
    unittest.main()
