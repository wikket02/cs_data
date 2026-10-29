"""Specified versus realized domain moments. MEANS and SDS in generate.py are the parameters (mu, sigma) of the normal
distribution BEFORE truncation to [0, 3]. The truncated variable's mean and SD therefore differ from them, and the
sample moments of 200 profiles differ again through sampling variation. Writes ../results/realized_moments.csv"""
import os, numpy as np, pandas as pd
from scipy import stats
from analysis import RES
from generate import DOMAINS, MEANS, SDS, SEEDS, generate_dataset, truncnorm_params
rows = []
data = {s: generate_dataset(s)[0] for s in SEEDS}
for d in DOMAINS:
    a, b = truncnorm_params(MEANS[d], SDS[d])
    tm, tv = stats.truncnorm.stats(a, b, loc=MEANS[d], scale=SDS[d], moments="mv")
    rows.append(dict(domain=d, param_mean=MEANS[d], param_sd=SDS[d], truncated_mean=float(tm), truncated_sd=float(np.sqrt(tv)),
                     seed42_mean=data[42][d].mean(), seed42_sd=data[42][d].std(ddof=1),
                     five_seed_mean=np.mean([data[s][d].mean() for s in SEEDS]),
                     five_seed_sd=np.mean([data[s][d].std(ddof=1) for s in SEEDS])))
out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "realized_moments.csv"), index=False)
print(out.round(3).to_string())
