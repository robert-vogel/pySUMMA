from unittest import TestCase, main

import numpy as np
from summa import decomposition as decomp


class TestMatrix(TestCase):
    m = 10
    max_iter = 100
    tol = 1e-6
    eig_val = 1
    eig_vec = np.arange(m)
    eig_vec = eig_vec / np.sqrt(np.sum(eig_vec**2))
    q = np.dot(eig_vec.reshape(m, 1), eig_vec.reshape(1, m))

    def test_init_assignments(self):

        mat = decomp.Matrix(self.max_iter, self.tol)

        self.assertEqual(mat.max_iter, self.max_iter)
        self.assertEqual(mat.tol, self.tol)

        mat = decomp.Matrix(tol=self.tol, max_iter=self.max_iter)

        self.assertEqual(mat.max_iter, self.max_iter)
        self.assertEqual(mat.tol, self.tol)

    def test_algorithm_update(self):
        """Input rank one matrix Q, then Q should be returned."""
        mat = decomp.Matrix(self.max_iter, self.tol)

        q = self.q.copy()
        q = q - np.diag(self.eig_vec**2)
        q_out, eig_val, eig_vec = mat._update_r(q, self.q)

        self.assertAlmostEqual(eig_val, self.eig_val)

        for i in range(self.m):
            self.assertAlmostEqual(eig_vec[i], self.eig_vec[i])

            for j in range(self.m):
                self.assertAlmostEqual(q_out[i, j], self.q[i, j])

    def test_algorithm_convergence(self):
        """Input rank one matrix Q, then Q should be returned."""

        mat = decomp.Matrix(self.max_iter, self.tol)

        eig_val, eig_vec, num_iters = mat._infer_matrix(self.q, False)

        self.assertEqual(num_iters, 2)
        self.assertAlmostEqual(eig_val, self.eig_val)

        for i in range(self.m):
            self.assertAlmostEqual(eig_vec[i], self.eig_vec[i])

    def test_errors_in_fit(self):
        mat = decomp.Matrix()

        with self.assertRaises(ValueError):
            mat.fit(np.eye(2))

        with self.assertRaises(ValueError):
            mat.fit(self.eig_vec)

        with self.assertRaises(ValueError):
            mat.fit(np.ones((4,4,4)))

        with self.assertRaises(ValueError):
            mat.fit(np.eye(4,5))

        with self.assertRaises(ValueError):
            q = np.identity(3)
            q[0,1] = 1

            mat.fit(q)

        with self.assertRaises(AttributeError):
            mat.fit(self.q.tolist())

    def test_fit_convergence(self):
        mat = decomp.Matrix()

        eig_val, eig_vec, num_iters = mat.fit(self.q)

        self.assertEqual(num_iters, 2)
        self.assertAlmostEqual(eig_val, self.eig_val)

        for i in range(self.m):
            self.assertAlmostEqual(eig_vec[i], self.eig_vec[i])


class TestTensor(TestCase):
    m = 4

    eig_vector = np.arange(1, m+1)
    eig_vector = eig_vector / np.sum(eig_vector**2)

    tensor = np.zeros(shape=(m,m,m))

    for i in range(m):
        for j in range(m):
            for k in range(m):
                tensor[i, j, k] = (eig_vector[i] * 
                                   eig_vector[j] * 
                                   eig_vector[k])

    idx_ijk = [(0,1,2), (0, 1, 3), (0, 2, 3), (1,2,3)]
    elements_ijk = []
    for idx in idx_ijk:
        elements_ijk.append(eig_vector[idx[0]] *
                          eig_vector[idx[1]] *
                          eig_vector[idx[2]])

    def test_init_assignments(self):
        t = decomp.Tensor()

    def test_get_tensor_idx_N_lt_k(self):
        """Test when the number of methods is less than 3"""
        t = decomp.Tensor()

        input_vals = [-2, 0, 1, 2]

        for m in input_vals:
            idx = t._get_tensor_idx(m)
            self.assertEqual(len(idx), 0)

    def test_get_tensor_idx(self):
        t = decomp.Tensor()

        idx = t._get_tensor_idx(3)
        expected_idx = [(0,1,2)]

        self.assertEqual(len(idx), len(expected_idx))
        for i, idx_val in enumerate(expected_idx[0]):
            self.assertEqual(idx[0][i], idx_val)


        idx = t._get_tensor_idx(self.m)

        self.assertEqual(len(idx), len(self.idx_ijk))
        for i, element in enumerate(self.idx_ijk):
            for j, idx_val in enumerate(element):
                self.assertEqual(idx[i][j], idx_val)

    def test_get_vals(self):
        t = decomp.Tensor()

        vec_vals, tensor_vals = t._get_vals(self.tensor, self.eig_vector)

        for i, true_val in enumerate(self.elements_ijk):
            self.assertEqual(vec_vals[i], true_val)
            self.assertEqual(tensor_vals[i], true_val)

    def test_fit_errors(self):
        t = decomp.Tensor()

        with self.assertRaises(AttributeError):
            t.fit(self.tensor, 1)

        with self.assertRaises(AttributeError):
            t.fit(self.tensor, self.eig_vector.tolist())

        with self.assertRaises(ValueError):
            t.fit(self.tensor, self.eig_vector[:2])

        with self.assertRaises(ValueError):
            t.fit(self.tensor,
                  self.tensor)
            
        with self.assertRaises(ValueError):
            t.fit(self.eig_vector, self.eig_vector)

        with self.assertRaises(ValueError):
            t.fit(np.ones(shape=(self.m, 2)),
                  self.eig_vector)

        with self.assertRaises(ValueError):
            t.fit(np.ones(shape=(self.m, self.m, self.m+1)),
                  self.eig_vector)

    def test_fit(self):
        t = decomp.Tensor()

        m = 3
        singular_val = t.fit(self.tensor[:m, :m, :m],
                             self.eig_vector[:m])

        self.assertAlmostEqual(singular_val, 1)

        singular_val = t.fit(self.tensor,
                             self.eig_vector)

        self.assertAlmostEqual(singular_val, 1)


if __name__ == "__main__":
    main()
