"""Sensitivity of the Stage 2 results to the noise fraction (sigma as a fraction of SD of the unadjusted weighted score).
Tests whether the choice of 0.58 drives the conclusions. Writes ../results/noise_sensitivity.csv"""
import os, pandas as pd, numpy as np
from analysis import full_seed, RES
from generate import SEEDS, DOMAINS
rows = []
for f in [0.30, 0.45, 0.58, 0.75, 1.00]:
    per = [full_seed(s, noise_frac=f) for s in SEEDS]
    rows.append(dict(
        noise_frac=f, theoretical_r2=1 / (1 + f ** 2),
        mean_observed_r2=np.mean([p["model"].rsquared for p in per]),
        mean_adj_r2=np.mean([p["model"].rsquared_adj for p in per]),
        mean_n_sig=np.mean([sum(p["model"].pvalues[d] < .05 for d in DOMAINS) for p in per]),
        top_domain_AC_seeds=sum(p["betas"].idxmax() == "AC" for p in per),
        h1_pass_seeds=sum(p["h1"]["rho"].min() >= .95 for p in per),
        h3_pass_seeds=sum(p["h3"]["rho"].min() >= .80 for p in per),
        h2_pass_seeds=sum(min(p["h2"]["rho_coef"].min(), p["h2"]["rho_priority"].min()) >= .80 for p in per),
        min_h2_rho_coef=min(p["h2"]["rho_coef"].min() for p in per)))
out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "noise_sensitivity.csv"), index=False)
print(out.round(3).to_string())
