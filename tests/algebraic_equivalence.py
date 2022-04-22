import unittest
import torch
import numpy as np
import random
import functools
import utils


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

    def test_vectorize_methods(self):

        size = (7, 11, 3)
        actual = torch.rand(size=size)

        vect = utils.vectorize(actual).flatten()
        expected = utils.unvectorize(vect, size=size).contiguous()

        self.assertEqualTensor(actual, expected)

    def test_mixed_kron_matrix_vector_prod(self):
        A = torch.rand(size=(5, 3))
        B = torch.rand(size=(4, 2))
        V = torch.rand(size=(3, 2))

        # print( torch.kron(B.transpose(0, 1).contiguous(), A), C.flatten() )
        true_vect = torch.matmul(torch.kron(B, A), V.transpose(0, 1).flatten())
        pred_vect = torch.matmul(torch.matmul(A, V), B.transpose(0, 1)).transpose(0, 1).flatten()

        self.assertEqualTensor(true_vect, pred_vect)

    # def test_mixed_kron_triple_matrix_vector_prod_one_side(self):
    #     '''
    #
    #     '''
    #
    #     B = torch.rand(size=(12, 9))
    #     C = torch.rand(size=(3, 3))
    #     D = torch.rand(size=(3, 4))
    #     V = torch.rand(size=(9, 4, 3))
    #
    #     v = utils.vectorize(V).flatten()
    #     true_vect = torch.kron(B, torch.kron(C, D)) @ v
    #
    #     pred_vect = utils.vectorize(torch.matmul(
    #             utils.vectorize(torch.matmul(torch.matmul(D.unsqueeze(0), V), C.transpose(0, 1).unsqueeze(0))).transpose(0, 1),
    #             B.transpose(0, 1)
    #         ))
    #
    #     self.assertEqualTensor(true_vect, pred_vect)

    # def test_mixed_kron_triple_matrix_vector_prod(self):
    #     '''
    #     Claim: vect(V)^t ( B \kron C \kron D ) vect(V) is equal to vec(V)^t vec( (DVC^t)B^t )
    #
    #     '''
    #
    #     B = torch.rand(size=(12, 9))
    #     C = torch.rand(size=(3, 3))
    #     D = torch.rand(size=(3, 4))
    #     V = torch.rand(size=(9, 4, 3))
    #
    #     v = utils.vectorize(V).flatten()
    #     true_vect = torch.matmul(torch.matmul(v, torch.kron(B, torch.kron(C, D))), v)
    #
    #     pred_vect = torch.matmul(
    #         v,
    #         utils.vectorize(torch.matmul(
    #             utils.vectorize(torch.matmul(torch.matmul(D.unsqueeze(0), V), C.transpose(0, 1).unsqueeze(0))).transpose(0, 1),
    #             B.transpose(0, 1)
    #         ))
    #     )
    #
    #     self.assertEqualTensor(true_vect, pred_vect)
    # def test_mixed_kron_triple_matrix_vector_prod(self):
    #     '''
    #     Claim: vect(V)^t ( B \kron C \kron D ) vect(V) is equal to vec(V)^t vec( (DVC^t)B^t )
    #
    #     '''
    #
    #     B = torch.randn(size=(11, 11))
    #     C = torch.randn(size=(3, 3))
    #     D = torch.rand(size=(4, 4))
    #     V = torch.rand(size=(11, 3, 4))
    #
    #     v = utils.vectorize(V).flatten()
    #     true_vect = torch.matmul(torch.matmul(v, torch.kron(B, torch.kron(C, D))), v)
    #
    #     print("x: ",  )
    #     # pred_vect = torch.matmul(
    #     #     v,
    #     #     utils.vectorize(torch.matmul(
    #     #         utils.vectorize(torch.matmul(torch.matmul(D.unsqueeze(0), V), C.transpose(0, 1).unsqueeze(0))).transpose(0, 1),
    #     #         B.transpose(0, 1)
    #     #     ))
    #     # )
    #     pred_vect = v @ utils.vectorize( utils.vectorize((D.unsqueeze(0) @ V.transpose(1, 2) @ C).transpose(1, 2)).t() @ B.t() )
    #     print(true_vect, pred_vect)
    #
    #     self.assertEqualTensor(true_vect, pred_vect)

    def test_mixed_kron_triple_matrix_vector_prod(self):
        '''
        Claim: vect(V)^t ( B \kron C \kron D ) vect(V) is equal to vec(V)^t vec( (DVC^t)B^t )

        '''

        B = torch.rand(size=(9, 9))
        C = torch.randn(size=(3, 3))
        D = torch.randn(size=(4, 4))
        V = torch.rand(size=(9, 3, 4))

        v = utils.vectorize(V.transpose(1, 2)).flatten()
        true_vect = torch.matmul(torch.matmul(v, torch.kron(B, torch.kron(C, D))), v)

        pred_vect = torch.matmul(
            v,
            utils.vectorize(torch.matmul(
                utils.vectorize(torch.matmul(torch.matmul(D.unsqueeze(0), V.transpose(1, 2)), C.transpose(0, 1).unsqueeze(0))).transpose(0, 1),
                B.transpose(0, 1)
            ))
        )

        self.assertEqualTensor(true_vect, pred_vect)

if __name__ == '__main__':
    unittest.main()
