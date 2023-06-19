"""Binary and rank based simulation tools

By: Robert Vogel
"""
import numbers

import numpy as np
from scipy.special import ndtri

from summa import stats

class Binary:
    """Simulate the binary predictions by an ensemble of base classifiers.

    Store input parameters, if tpr and tnr are not specified in 
    the 'rates' argument then sample according to ba_lims.
    
    Args:
        M: (int) methods
        N: (int) samples
        N1: (int) positive class samples
        ba_lims : (float min_ba, float max_ba), such that each value
            v is 0 <= v <= 1
        rates : (dict, default None) dict keys value pairs of 
            ((M,) ndarray) of true positive rates, and 
            ((M,) ndarray) of true negative rates
        rng: 
    """
    def __init__(self, M, N, N1, 
                 ba_lims=(0.35, 0.9),
                 rates = None,
                 rng = None):
        
        self.M, self.N, self.N1 = M, N, N1

        self.rng = np.random.default_rng(rng)

        if rates is None:
            self.ba, self.tpr, self.tnr = self._sample_rates_from_ba_lims(ba_lims)

        elif isinstance(rates, dict):
            for j in range(self.M):
                if rates["tpr"][j] < 0 or rates["tpr"][j] > 1:
                    raise ValueError("Each tpr must be between 0 and 1")
                
                if rates["tnr"][j] < 0 or rates["tnr"][j] > 1:
                    raise ValueError("Each tnr must be between 0 and 1")
            
            self.tpr = rates["tpr"]
            self.tnr = rates["tnr"]

        else:
            raise TypeError("Rates must be a dictionary with keys\n "
                             "'tpr' and 'tnr' and the values as M "
                             "length\nnumpy arrays of rates.")
            
        self.labels = np.hstack([np.ones(N1), 
                                 -np.ones(N-N1)])

    @staticmethod
    def _sample_tpr_given_ba(ba):
        """Uniformly sample true positive rate and balanced accuracy.

        Recall that by definitions of true positive rate (tpr),
        true negative rate (tnr), and balanced accuracy (ba): 

        tpr = 2*ba - tnr
    
        define tpr_min = 0, and tpr_max = minimum (1, 2 * ba).
    
        Args:
            ba : ndarray
            (M,) length ndarray of balanced accuracies
    
        Returns:
            tpr : ndarray
            (M, ) length ndarray of randomly sampled true positive rates
        """
        # inititate tpr array
        tpr = np.zeros(ba.size)
    
        for j in range(ba.size):
            tpr_min = np.max([0, 2 * ba[j] -1])
    
            tpr_max = np.min([1, 2 * ba[j]])
            
            if tpr_min > tpr_max:
                raise ValueError("TPR min > TPR max")
            
            tpr[j] = self.rng.rand() * (tpr_max - tpr_min) + tpr_min
    
        return tpr

    def _sample_rates_from_ba_lims(self, ba_lims):
        """
        Uniformly sample balanced accuracy (ba) values for each of the
        M base classifiers.  Sample true positive rates (tpr), and 
        compute true negative rates from the sampled (BA) values.

        Args:
            ba_lims : [float, float] with lower bound ba_lims[0]
                and upper bound (ba_lims[1]) 

        Returns:
            ba: ((M,) ndarray) of balanced accuracies
            tpr ((M,) ndarray) of true positive rates
            tnr :((M,) ndarray) of true negative rates
        """
        # check that ba_lims is:
        #    1) that the first entry is less than the second,
        #    2) that each entry is between [0, 1]
        
        if ba_lims[0] >= ba_lims[1]:
            raise ValueError("ba_lims[0] must be less than ba_lims[1]")

        elif ba_lims[0] < 0 or ba_lims[1] > 1:
            raise ValueError("B.A. limits must be between 0 and 1")

        delta_ba = np.max([0, ba_lims[1] - ba_lims[0]])

        ba = self.rng.rand(self.M) * delta_ba + ba_lims[0]

        tpr = self._sample_tpr_given_ba(ba)

        tnr = 2*ba - tpr
        return [ba, tpr, tnr]

    @property
    def ba(self):
        """Compute the Balanced Accuracy from TPR and TNR.

        Returns:
            The balanced accuracies of M base classifiers ((M,) ndarray)
        """
        return 0.5*(self.tpr + self.tnr)

    def sim(self):
        """Generate simulation data, and store as class properties.
        
        Generated properties:
        self.data : ndarray
        (M, N) ndarray of binary [-1, 1] predictions

        
        self.data : ndarray
        (M, N) ndarray of M binary classifier binary predictions of N samples
        """
        # initialize ndarrays
        self.data = np.zeros(shape=(self.M, self.N))

        # generate samples for each classifier
        for j in range(self.M):
            
            # loop over samples
            for i in range(self.N):
                # generate random number u between 0 and 1
                u = self.rng.rand()
                
                # if true sample label is positive, then sample
                # from true positive rate
                if self.labels[i] == 1:
                    self.data[j, i] = 1 if u <= self.tpr[j] else -1
                # if samples are not from the positive class, 
                # they are from negative class
                else:
                    self.data[j, i] = -1 if u <= self.tnr[j] else 1


class EnsembleGaussianPredictions:
    """Model block diagonally correlated base classifier predictions.

    Fraction of classifiers assigned to each correlated group
    correlation per grup
    auc per classifier

    Assume that the number of classifiers is >= 2

    Args: 
        m_cls: None or int representing number of classifiers
        auc: float or ((auc_min, auc_max) array) or 
            ((auc_bc_1, auc_bc_2, ..., auc_bc_m_cls), array).  Each
            auc value is the area under receiver operating characteristic and
            consequently must be a value on the interval [0,1]
        group_sizes: ((G groups,) tuple, list, or np.ndarray) 
        group_corrs: (float or (G Pearson corr coefs,) tuple, list, or np.ndarray)

    Example:

        Generate 10 conditionally indepedent base classifiers with AUC of 0.7

        >>> EnsembleGaussianPredictions(auc = 0.7, m_cls = 10)

        Generate 15 conditionally independent base classifiers uniformally
        distributed on the interval [0.4, 0.75]

        >>> EnsembleGaussianPredictions(auc=(0.4, 0.75), m_cls = 15)

        Generate 10 base classifiers with AUC on the range [0.4, 0.75], such 
        that base classifiers 1-4, 5-6, and 7-10 exhibit class conditional 
        correlation of 0.6, 0.7, and 0, respectively.

        >>> EnsembleGaussianPredictions(auc=(0.4, 0.75),
                                        m_cls = 15,
                                        group_sizes=(4, 2, 4),
                                        group_corrs=(0.6, 0.7, 0))

        Generate 4 base classifiers with AUC [0.8, 0.63, 0.713], such 
        that base classifiers 1-2 and 3 exhibit class conditional 
        correlation of 0.6 and 0, respectively.

        >>> EnsembleRankPredictions(auc=(0.8, 0.63, 0.713),
                                    group_sizes=(2, 1),
                                    group_corrs=(0.6, 0))

        or equivalently,

        >>> EnsembleRankPredictions(auc=(0.8, 0.63, 0.713),
                                    group_sizes=(2,),
                                    group_corrs=(0.6,))
    """
    def __init__(self,
            m_cls=None,
            auc=None,
            group_corrs=None,
            group_sizes=None, 
            seed=None):

        self.m_cls, self.auc = self._parse_auc(auc, 
                m_cls)

        self._cov = self._mk_covariance_matrix(group_corrs, group_sizes)
        self._delta = self._mk_delta_array(self.auc)

        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        return self.m_cls

    def _parse_auc(self, auc, m_cls):
        # TypeError if m_cls is not int
        if isinstance(auc, float):
            auc = [auc for _ in range(m_cls)]

        # if given auc array represents each base classifier in 
        # the ensemble
        if m_cls is None:  
            m_cls = len(auc)

        # Verify AUC value is within AUC range
        for auc_i in auc:
            if auc_i < 0 or auc_i > 1: raise ValueError

        # if given m_cls was neither None or int, then error
        # thrown here. 
        if m_cls == len(auc): 
            return m_cls, auc
        elif len(auc) == 2 and auc[0] < auc[1]:
            return m_cls, np.linspace(auc[0], auc[1], m_cls)

        raise ValueError

    def _mk_delta_array(self, auc):
        """Compute delta array from auc values.
        
        Delta for each base classifer i is computed by

        \Delta_i = \sqrt{cov_i|positive_class + cov_i|negative_class} *
                        inv_standard_normal_cumulative(auc_i)

        as described in Marzben, C. "The ROC Curve and the Area under It
        as Performance Measures", Weather and Forecasting 2004.
        """
        delta = np.zeros(len(self))

        for i, auc_val in enumerate(auc):
            delta[i] = np.sqrt(2 * self._cov[i,i])  * ndtri(auc_val)

        return delta


    def _mk_covariance_matrix(self, group_corrs, group_sizes):
        """Make conditional covariance matrix.
        
        Parse input arguments and construct covariance matrix.  I have
        limited the covariance matrix to be the Pearson correlation matrix.
        Therefore, the diagonal elements will always be one, the matrix
        symmetric, and off-diagonal elements on the interval [-1, 1].
        """

        # either group_corrs and group_sizes are specified or they
        # are both not specified.  
        if group_corrs is None and group_sizes is None:
            return np.eye(len(self))
        elif group_corrs is None and group_sizes is not None:
            raise TypeError
        elif group_corrs is not None and group_sizes is None:
            raise TypeError
        elif np.sum(group_sizes) > self.m_cls:
            raise ValueError
        elif (not isinstance(group_corrs, numbers.Number) and 
                len(group_corrs) != len(group_sizes)):
            raise ValueError

        if isinstance(group_corrs, numbers.Number):  # if each group has same corr
            group_corrs = [group_corrs for g in group_sizes]

        cov = np.zeros(shape=(len(self), len(self)))

        g_start = 0
        for m, corr in zip(group_sizes, group_corrs):

            if corr > 1 or corr < -1:
                raise ValueError
            elif not isinstance(m, numbers.Number):
                raise TypeError

            g_final = g_start + m


            # loop over cov elements of group and set
            # the covariance values as specified
            for i in range(g_start, g_final):
                for j in range(g_start, g_final):

                    cov[i, j] = 1 if i == j else corr

            g_start = g_final

        # any remaining base classifiers are assumed 
        # conditionally independent,
        for i in range(g_final, len(self)):
            cov[i, i] = 1
        
        return cov

    @property
    def size(self):
        return self.m_cls

    def sample(self, label):
        """A single sample base classifier predictions."""
        s = self.rng.multivariate_normal(np.zeros(self._cov.shape[0]),
                        self._cov, size=1)

        if label ==1:
            s += self._delta

        return s


class EnsembleRankPredictions(EnsembleGaussianPredictions):
    def sample(self, n_samples, n_positive_class):
        scores = np.zeros(shape=(self.m_cls, n_samples))
        labels = np.zeros(n_samples)
        labels[:n_positive_class] = 1

        for j in range(n_samples):
            scores[:, j] = super().sample(labels[j])

        return stats.rank_transform(scores), labels