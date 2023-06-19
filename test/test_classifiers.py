"""Test classifiers

By Robert Vogel
"""
from unittest import TestCase, main

import numpy as np
from summa import classifiers as cls


class TestEnsembleAbc(TestCase):
    n_samples, m_cls = 1000, 7
    n_positive = 300

    data = np.arange(n_samples).reshape(1,n_samples) + 1
    data = np.tile(data, (m_cls, 1))

    def get_labels(self, positive_encoding, negative_encoding):
        return np.hstack([np.full(self.n_positive, positive_encoding),
            np.full(self.n_samples - self.n_positive, negative_encoding)])

    def set_up_ens_params(self):
        abc = cls.EnsembleABC()
        abc.n_samples, abc.m_cls = self.n_samples, self.m_cls
        abc.weights = np.ones(self.m_cls) / self.m_cls
        abc._prevalence = self.n_positive / self.n_samples
        return abc

    def test_init_assignments(self):
        abc = cls.EnsembleABC()

        self.assertIsNone(abc.n_samples)
        self.assertIsNone(abc.m_cls)
        self.assertIsNone(abc._positive_class)
        self.assertIsNone(abc._negative_class)
        self.assertIsNone(abc._prevalence)
        self.assertIsNone(abc.weights)

    def test_name(self):
        abc = cls.EnsembleABC()

        self.assertEqual(abc.name, "EnsembleABC")

    def test_fit_raises(self):
        abc = self.set_up_ens_params()

        with self.assertRaises(NotImplementedError):
            abc.fit()

        with self.assertRaises(NotImplementedError):
            abc.fit(1,2,3,4,5)

        with self.assertRaises(NotImplementedError):
            rng = np.random.default_rng()
            abc.fit(rng.random(size=(5, 100)))

    def test_get_scores_raises(self):
        abc = self.set_up_ens_params()

        with self.assertRaises(AttributeError):
            abc.get_scores(self.data.tolist())

        with self.assertRaises(AttributeError):
            abc.get_scores(3)

        with self.assertRaises(ValueError):
            abc.get_scores(self.data[:, 0])

        with self.assertRaises(ValueError):
            abc.get_scores(np.vstack([self.data, self.data]))

    def test_get_scores(self):
        abc = self.set_up_ens_params()

        scores = abc.get_scores(self.data)

        for i, s in enumerate(scores):
            self.assertAlmostEqual(s, i+1)

    def test_inference_raises(self):
        abc = self.set_up_ens_params()

        with self.assertRaises(AttributeError):
            abc.get_scores(self.data.tolist())

        with self.assertRaises(AttributeError):
            abc.get_scores(3)

        with self.assertRaises(ValueError):
            abc.get_scores(self.data[:, 0])

        with self.assertRaises(ValueError):
            abc.get_scores(np.vstack([self.data, self.data]))

    def test_inference_1(self):
        """Test inference for 1,0 class encodings"""
        abc = self.set_up_ens_params() 

        abc._positive_class = 1
        abc._negative_class = 0


        # all samples should be positive class
        inferred_labels = abc.get_inference(self.data)

        for l in inferred_labels:
            self.assertEqual(l, abc._positive_class)

        # all samples should be negative
        data = self.data.copy()
        data = - data

        inferred_labels = abc.get_inference(data)

        for l in inferred_labels:
            self.assertEqual(l, abc._negative_class)

        # the number of positive class samples should be n_positive

        data = self.data.copy()
        data = self.n_positive - data

        inferred_labels = abc.get_inference(data)
        true_labels = self.get_labels(abc._positive_class,
                abc._negative_class)

        for il, tl in zip(inferred_labels, true_labels):
            self.assertEqual(il, tl)

    def test_inference_2(self):
        """Test inference for 1,-1 class encodings"""
        abc = self.set_up_ens_params()

        abc._positive_class = 1
        abc._negative_class = -1

        # all samples should be positive class
        inferred_labels = abc.get_inference(self.data)

        for l in inferred_labels:
            self.assertEqual(l, abc._positive_class)

        # all samples should be negative
        data = self.data.copy()
        data = - data

        inferred_labels = abc.get_inference(data)

        for l in inferred_labels:
            self.assertEqual(l, abc._negative_class)

        # the number of positive class samples should be n_positive

        data = self.data.copy()
        data = self.n_positive - data

        inferred_labels = abc.get_inference(data)
        true_labels = self.get_labels(abc._positive_class,
            abc._negative_class)

        for il, tl in zip(inferred_labels, true_labels):
            self.assertEqual(il, tl)

# TODO
class TestSummaAbc(TestCase):
    pass

# TODO
class TestSumma(TestCase):
    pass

# TODO
class TestRankWoc(TestCase):
    pass

# TODO
class TestBinaryWoc(TestCase):
    pass

# TODO
class TestSml(TestCase):
    pass




if __name__ == "__main__":
    main()
