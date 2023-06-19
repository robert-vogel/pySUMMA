"""Matrix and tensor decompositions.

By: Robert Vogel
"""

import numpy as np
from scipy.special import comb

class Matrix:
    """Implement the iterative inference from Ahsen et al. [2].

    It has been shown that the covariance matrix of conditionally 
    independent binary [1] and rank [2] base classifier predictions
    has a special structure.  This is that the off-diagonal 
    elements i,j are proportional to the product of the i^th and 
    j^th base classifier's performance as measured by the balanced 
    accuracy [1] or the difference of sample class conditioned 
    average rank predictions [2].  Consequently, these entries are 
    those of a rank one matrix formed by the outer product of a 
    vector encoding base classifier performances.

    In this module we infer the performance vector from the 
    empirical covariance matrix by implementing the iterative 
    procedue of Ahsen et al. [2].

    References:
        1. F. Parisi, et al. Ranking and combining multiple 
            predictors without labeled data. Proceedings of the 
            National Academy of Sciences, 111(4):1253--1258, 2014.
        
        2. M.E. Ahsen, R. Vogel, and G. Stolovitzky. Unsupervised 
            evaluation and weighted aggregation of ranked 
            predictions. arXiv preprint arXiv:1802.04684, 2018.

    Args:
        max_iter: max number of iterations (int, default 5000)
        tol: stopping threshold (float, default 1e-6)
    """
    def __init__(self, max_iter = 5000, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _update_r(Q_off_diagonal, R):
        '''Update the estimate of the rank one matrix.
    
        Args:
            Q_off_diagonal: Covariance matrix with diagonal 
                entries set to zero ((M, M) ndarray)
            R: estimated Rank one matrix ((M, M) ndarray)
    
        Returns:
            ((M, M) ndarray) Updated estimate of the rank one matrix R, 
                
            (float) eigenvalue of R 
            ((M,) ndarray) eigenvector of R
        '''
        l, v = np.linalg.eigh(R)

        # compute the diagonal of a rank one matrix
        rdiag = np.diag(l[-1] * v[:, -1]**2)

        return (Q_off_diagonal + rdiag, l[-1], v[:, -1])

    def _infer_matrix(self, Q, return_iters = False):
        """Algorithm for inferring performance vector.

        Algorithm for inferring the diagonal entries which would make
        the covariance matrix Q, of full rank, a rank one matrix.
    
        Args:
            Q: ((M, M) ndarray) covariance matrix
            return_items: (bool) return values for each iteration? 
                (default False)
    
        Returns:
            return_items = False:
                (float) rank one matrix eigenvalue
                (ndarray) eigenvector
                (int) iterations until convergence
            return_items = True:
                (float) rank one matrix eigenvalue (float),
                (ndarray) eigenvector (ndarray),
                (ndarray) the inferred eigenvalue at each iteration, 
                (int) interest until convergence
                
        Raises:
            RuntimeError: raised when convergence criteria not met.
        """
        Q_off_diagonal = Q - np.diag(np.diag(Q))
        R = Q.copy()

        j = 0

        epsilon = np.sum(2*np.diag(Q))
        eig_values = [epsilon]

        while epsilon > self.tol and j < self.max_iter:
            # decompose the rank one approximation
            R, eig_value, eig_vector = self._update_r(Q_off_diagonal, R)

            epsilon = np.abs(eig_values[-1] - eig_value)

            eig_values += [eig_value]
            j += 1

        if j == self.max_iter:
            raise RuntimeError(("Matrix decomp. did not converge:\n"
                                "a) Increase the maximum number of"
                                f" iterations above {self.max_iter:d}, or \n"
                                "b) increase the minimum"
                                f" tolerance above {self.tol:.4f}."))

        # Assume that the majority of methods
        # correctly rank samples according to latent class
        # consequently the majority of Eigenvector
        # elements should be positive.
        if np.sum(eig_vector < 0) > Q.shape[0]/2:
            eig_vector = -eig_vector

        if return_iters:
            return (eig_value, eig_vector, eig_values[1:], j)
        else:
            return (eig_value, eig_vector, j)

    def fit(self, Q):
        """Find the diagonal entries that make Q a rank one matrix.

        Args:
            Q: The covariance matrix ((M, M) ndarray)
        
        Returns:
            eig_value: (float) 
            eig_vector: (ndarray) 
            num_iters: (int) 
        """
        if Q.shape[0] < 3:
            raise ValueError(("The minimum required number of "
                            "base classifiers is 3."))
        elif Q.ndim != 2:
            raise ValueError(("Input ndarray must be a matrix "
                            "(ndim == 2)."))
        elif not np.array_equal(Q, Q.T):
            raise ValueError("Not a symmetric matrix.")

        return self._infer_matrix(Q)


class Tensor:
    """ Fit singular value of third central moment tensor.
    
    Fit the singular value (l) for the rank one tensor whose elements
    T_{i, j, k} = l * v[i] * v[j] * v[k] where i \neq j \neq k and v is 
    the Eigenvector from the covariane decomposition.
    """
    @staticmethod
    def _get_tensor_idx(M):
        """Get indecies i \neq j \neq k of the squre 3rd order tensor.
    
        Args:
            M: The number of entries of the tensor of interest (integer)
    
        Returns:
            idx : list of tuples containing indexes (i, j, k) 
                such that i \neq j \neq k (list)
        """
        idx = list(range(comb(M, 3, exact=True)))

        l = 0
        for i in range(M-2):
            for j in range(i+1, M-1):
                for k in range(j+1, M):
                    idx[l] = (i, j, k)
                    l += 1
        return idx

    def _get_vals(self, third_moment_tensor, eig_vec):
        """Extract tensor elements T_{ijk} for i \neq j \neq k.
        
        Then what is returned are n predicted and actual third 
        central moment elements for i \neq j \neq k.

        Args:
            third_moment_tensor: ((M,M,M) ndarray) third central 
                moment tensor 
            eig_vec: ((M,) ndarray)
        
        Returns:
            eig_vals : ((n,) ndarray) the expected third
                central moment tensor elements computed from the
                covariance Eigenvector
            tensor_vals: ((n,) ndarray) tensor elements 
                i \neq j \neq k 
        """
        tensor_idx = self._get_tensor_idx(eig_vec.size)

        tensor_vals = np.zeros(len(tensor_idx))
        eig_vals = np.zeros(len(tensor_idx))

        for i, t_idx in enumerate(tensor_idx):
            tensor_vals[i] = third_moment_tensor[t_idx[0],
                                                 t_idx[1],
                                                 t_idx[2]]

            eig_vals[i] = (eig_vec[t_idx[0]] *
                            eig_vec[t_idx[1]] *
                            eig_vec[t_idx[2]])

        return (eig_vals, tensor_vals)

    def fit(self, third_moment_tensor, eig_vec):
        """Fit third momenet tensor singular value by linear regression.

        Args:
            third_moment_tensor: ((M,M,M) ndarray) third central 
                moment tensor 
            eig_vec: ((M,) ndarray)
        """

        M = eig_vec.size

        if M < 3:
            raise ValueError("The minimum required number of base "
                             "classifiers is 3.")
        elif eig_vec.ndim != 1:
            raise ValueError("Input vector should be an M "
                              "length ndarray")
        elif third_moment_tensor.shape != (M, M, M):
            raise ValueError("Input tensor should be an MxMxM and"
                             "input vector M length ndarray")

        eig_vals, tensor_vals = self._get_vals(third_moment_tensor,
                                               eig_vec)

        if eig_vals.size == 1:

            return tensor_vals[0] / eig_vals[0]

        dy = np.sum(eig_vals * tensor_vals)
        dx = np.sum(eig_vals**2)

        return dy / dx
