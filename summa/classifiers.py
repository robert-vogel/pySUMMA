"""Classifiers for binary classification.

By: Robert Vogel
"""

import numpy as np
from . import decomposition as decomp
from . import stats


class EnsembleABC:
    """Base class for ensemble methods of summa package."""
    def __init__(self):
        self.n_samples = None                   # num samples
        self.m_cls = None                   # num base classifiers
        self._positive_class = None
        self._negative_class = None
        self._prevalence = None
        self.weights = None

    @property
    def name(self):
        return self.__class__.__name__

    def fit(self, *_):
        raise NotImplementedError

    def get_scores(self, data):
        """Compute scores by weighted sum

        Args:
            data : ((m, n) ndarray) 

        Returns:
           s: ((n,) ndarray) scores
        """
        if data.ndim != 2:
            raise ValueError("Requsite input ((m, n) ndarray), not " 
                              f"{data.ndim}")
        if data.shape[0] != self.m_cls:
            raise ValueError("Input sample does not consist "
                              f"of predictions by {self.m_cls} methods")

        s = 0
        for j, w in enumerate(self.weights):
            s += w * data[j, :]
        return s

    def get_inference(self, data):
        """Estimate class labels from scores.

        Args:
            data: ((m, n) ndarray) 

        Returns:
            labels: ((n,) ndarray) inferred sample class labels
        """
        labels = self.get_scores(data)
        labels[labels >= 0] = self._positive_class
        labels[labels < 0] = self._negative_class
        return labels


class SummaABC(EnsembleABC):
    def __init__(self, prevalence=None):
        super().__init__()

        self._prevalence = prevalence
        self._eig_val = None
        self._eig_vec = None
        self._num_iters = None
        self._tensor_sv = None

    @property
    def delta_norm(self):
        '''Norm of performance vector Delta.

        Case 1: known prevalence
            Return the norm of delta using the a priori 
            specified positive class prevalence.
        Case 2: inferred prevalence
            Return the inferred norm by using
            the tensor and covariance singular values.
        
        Returns: 
            The norm of the performance vector, Delta (float).
        '''
        if self._prevalence is None:
            beta = (self._tensor_sv / self._eig_val)**2
            return np.sqrt(beta + 4*self._eig_val)

        return np.sqrt(self._eig_val / 
                       (self._prevalence*(1-self._prevalence)))

    @property
    def delta(self):
        """Inferred performance vector, Delta."""
        return self._eig_vec * self.delta_norm

    @property
    def auc(self):
        """Inferred AUC."""
        return self.delta / self.n_samples + 0.5

    @property
    def prevalence(self):
        """Either infer or return the postive class prevalence."""

        if self._prevalence is None:
            beta = self._tensor_sv / self._eig_val
            return 0.5 + 0.5*beta / self.delta_norm

        return self._prevalence


class Summa(SummaABC):
    """Apply SUMMA ensemble to data.

    Args:
        prevalence: the fraction of samples from the positive class.
            When None, infer prevalence from data. 
            (float [0,1] or None, default None)
    """
    def get_scores(self, data):
        stats.is_rank(data)

        s = super().get_scores(data)

        return stats.mean_rank(data.shape[1]) - s

    def fit(self, data, tol=1e-3, max_iter=500):
        """Infer SUMMA weights from unlabeled data.

        Args:
            data: ((M, N) ndarray) matrix of N sample rank predictions 
                by M base classifiers
            tol: (float) the tolerance for convergence in matrix 
                decomposition (default 1e-3).
            max_iter: (int) the maximum number of iterations for 
                matrix decomposition (default 500).
        """
        stats.is_rank(data)

        self.m_cls, self.n_samples = data.shape

        self._negative_class = 0
        self._positive_class = 1
        
        mat = decomp.Matrix(tol=tol, max_iter=max_iter)
        self._eig_val, self._eig_vec, self._num_iters = mat.fit(np.cov(data))

        if self._prevalence is None:
            tensor = decomp.Tensor()
            self._tensor_sv = tensor.fit(
                    stats.third_central_moment(data),
                    self._eig_vec)
        
        self.weights = self._eig_vec


class RankWoc(EnsembleABC):
    """WOC for ranked based classifier predictions.

    Args:
        prevalence: (float) the fraction of samples from the
            positive class.
            When None given, infer prevalence from data. 
    """
    def get_scores(self, data):
        if stats.is_rank(data):
            s = super().get_scores(data)
            return stats.mean_rank(data.shape[1]) - s

    def fit(self, data):
        self.m_cls, self.n_samples = data.shape

        self._positive_class = 1
        self._negative_class = 0

        self.weights = np.full(self.m_cls, 1 / self.m_cls)


class BinaryWoc(EnsembleABC):
    """WOC for binary (-1, 1) classifier predictions.

    Args:
        prevalence: 
    """
    def fit(self, data):
        self.m_cls, self.n_samples = data.shape

        self._positive_class = 1
        self._negative_class = -1

        self.weights = np.full(self.m_cls, 1 / self.m_cls)
        
    def get_scores(self, data):
        if stats.is_binary(data):
            return super().get_scores(data)



class Sml(EnsembleABC):
    """Apply SML ensemble to data.
    
    Args:
        prevalence: (float) prevalence of positive class samples (default None)
    """

    
    def __init__(self, prevalence=None):
        super().__init__()

        self._prevalence = prevalence
        self._eig_val = None
        self._eig_vec = None
        self._num_iters = None
        self._tensor_sv = None

    @property
    def ba_norm(self):
        '''Compute the norm of the balanced accuracy.

        Case 1: known prevalence
            Return the norm of the performance vector using the a priori
            specified positive class prevalence and Eigenvalue from the
            matrix decomposition.
        Case 2: tensor singular value
            Return the inferred norm of the performance vector using the 
            Eigenvalue and singular value estimated from the matrix and tensor 
            decomposition, respectively.

        Returns:
            float, norm of the performance vector.
        '''
        if self._prevalence is None:
            beta = (self._tensor_sv / self._eig_val)**2
            return 0.5 * np.sqrt(beta + 4*self.cov.eig_value)
        else:
            return np.sqrt(self._eig_val / 
                           (4 * self._prevalence * (1-self._prevalence)))

    @property
    def ba(self):
        """Infer balanced accuracy of each base classifier."""
        return 0.5*(1 + self._eig_vec * self.ba_norm)

    @property
    def prevalence(self):
        """Return sample class prevalence."""
        if self._prevalence is None:
            beta = self._tensor_sv / self._eig_val
            return 0.5 * (1  - 0.5*beta / self.ba_norm)
        else:
            return self._prevalence

    def get_scores(self, data):
        if not stats.is_binary(data):
            return None

        return super().get_scores(data)

    def fit(self, data, tol=1e-3, max_iter=500):
        """Fit the SML model to the empirical covariance and third central moment.

        Args:
            data : ((m, n) ndarray) 
            tol: (float) the tolerance for matrix decomposition (default 1e-3).
            max_iter: (int) maximum number of iterations for matrix decomposition
                (default 500).
        """
        if not stats.is_binary(data):
            return None

        self.m_cls, self.n_samples = data.shape

        self._negative_class = -1
        self._positive_class = 1
        
        mat = decomp.Matrix(tol=tol, max_iter=max_iter)
        self._eig_val, self._eig_vec, self._num_iters = mat.fit(np.cov(data))

        if self._prevalence is None:
            tensor = decomp.Tensor()
            self._tensor_sv = tensor.fit(
                    stats.third_central_moment(data),
                    self._eig_vec)

        self.weights = self._eig_vec