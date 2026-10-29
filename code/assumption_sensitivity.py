"""Tests whether researcher-defined distribution parameters and domain weights drive the conclusions.
Each scenario regenerates all five seeded datasets and repeats the full Stage 2 / H1 / H2 / H3 analysis.
Output: ../results/assumption_sensitivity.csv"""
import os, numpy as np, pandas as pd
from analysis import full_seed, RES
from generate import SEEDS, DOMAINS, MEANS, SDS, WEIGHTS
w_rev = dict(zip(DOMAINS, [WEIGHTS[d] for d in reversed(DOMAINS)]))   # AC gets RA's weight, etc.
scen = {
  "Baseline (reported)": {},
  "All domain means -0.25": dict(means={d: m - .25 for d, m in MEANS.items()}),
  "All domain means +0.25": dict(means={d: m + .25 for d, m in MEANS.items()}),
  "All domain SDs x 0.75": dict(sds={d: s * .75 for d, s in SDS.items()}),
  "All domain SDs x 1.25": dict(sds={d: s * 1.25 for d, s in SDS.items()}),
  "Identical domain distributions (mean 1.25, SD 0.60)": dict(means={d: 1.25 for d in DOMAINS}, sds={d: .60 for d in DOMAINS}),
  "Equal domain weights (1/6 each)": dict(weights={d: 1 / 6 for d in DOMAINS}),
  "Reversed domain weights (RA .25 ... AC .10)": dict(weights=w_rev),
}
rows = []
for name, g in scen.items():
    per = [full_seed(s, **g) for s in SEEDS]
    rows.append(dict(
        scenario=name,
        mean_adj_r2=np.mean([p["model"].rsquared_adj for p in per]),
        top_domain_by_seed=",".join(p["betas"].idxmax() for p in per),
        top_is_AC_seeds=sum(p["betas"].idxmax() == "AC" for p in per),
        mean_n_sig=np.mean([sum(p["model"].pvalues[d] < .05 for d in DOMAINS) for p in per]),
        h1_pass_seeds=sum(p["h1"]["rho"].min() >= .95 - 1e-9 for p in per),
        h3_pass_seeds=sum(p["h3"]["rho"].min() >= .80 - 1e-9 for p in per),
        h2_pass_seeds=sum(min(p["h2"]["rho_coef"].min(), p["h2"]["rho_priority"].min()) >= .80 - 1e-9 for p in per)))
out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "assumption_sensitivity.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(out.round(3).to_string())
