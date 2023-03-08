"""Simulate data, fit, and plot results.

By: Robert Vogel
"""

import os
import numpy as np

from summa import classifiers as cls
from summa import simulate, plot, stats


m_classifiers = 15
auc = (0.4, 0.75)

n_samples = 2500
prevalence = 0.3
n_positive_samples = int(prevalence * n_samples)

sim = simulate.EnsembleRankPredictions(m_cls=m_classifiers,
        auc=auc)
data, labels = sim.sample(n_samples, n_positive_samples)

cl_summa = cls.Summa()
cl_summa.fit(data)

with open("summa_inferred_prevalence.txt", "w") as fid:
    fid.write("Positive Class Prevalence\n"
            f"Inferred:\t{cl_summa.prevalence:0.3f}\n"
            f"True:\t\t{prevalence:0.3f}")


cl_woc = cls.RankWoc()
cl_woc.fit(data)

roc = {}
_, _, roc["Woc"] = stats.roc(cl_woc.get_scores(data), labels)
_, _, roc["Summa"] = stats.roc(cl_summa.get_scores(data), labels)

savename = os.path.join(os.path.dirname(__file__),
            "summa_auc.png")

plot.performance(sim.auc, cl_summa.auc, roc, savename=savename)
