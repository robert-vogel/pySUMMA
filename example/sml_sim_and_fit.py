
"""Simulate data, fit, and plot results.

By: Robert Vogel
"""

import os
import numpy as np

from summa import classifiers as cls
from summa import simulate, plot, stats

m_classifiers = 15
ba_lims = (0.4, 0.75)

n_samples = 2500
prevalence = 0.3
n_positive_samples = int(prevalence * n_samples)

seed = 42
rng = np.random.default_rng(seed)

sim = simulate.Binary(m_classifiers, 
        ba_lims=ba_lims,
        seed=rng)
        
data, labels = sim.sample(n_samples, n_positive_samples)

cl_sml = cls.Sml()
cl_sml.fit(data)

with open("sml_inferred_prevalence.txt", "w") as fid:
    fid.write("Positive Class Prevalence\n"
            f"Inferred:\t{cl_sml.prevalence:0.3f}\n"
            f"True:\t\t{prevalence:0.3f}")


cl_woc = cls.BinaryWoc()
cl_woc.fit(data)

ba = {}
ba["Woc"] = stats.ba(cl_woc.get_inference(data), labels)
ba["Summa"] = stats.ba(cl_summa.get_inference(data), labels)

savename = os.path.join(os.path.dirname(__file__),
            "sml_ba.png")

plot.performance(sim.ba, cl_sml.ba, ba, savename=savename)
