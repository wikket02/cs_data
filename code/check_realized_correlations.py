"""Compares realized sample correlations in each seeded dataset with the NORTA targets. Output: ../results/realized_correlations.csv"""
import os, numpy as np, pandas as pd
from generate import generate_dataset, DOMAINS, NOISE_FRAC, SEEDS
from analysis import RES
within = [("AC", "IA"), ("CM", "AA"), ("IR", "RA")]; targets = {("AC", "IA"): .63, ("CM", "AA"): .59, ("IR", "RA"): .52}
rows = []
for s in SEEDS:
    c = generate_dataset(s, noise_frac=NOISE_FRAC)[0][DOMAINS].corr()
    bt = [c.loc[a, b] for i, a in enumerate(DOMAINS) for b in DOMAINS[i + 1:] if (a, b) not in within]
    rows.append(dict(seed=s, AC_IA=c.loc["AC", "IA"], CM_AA=c.loc["CM", "AA"], IR_RA=c.loc["IR", "RA"], between_mean=np.mean(bt), between_min=min(bt), between_max=max(bt)))
r = pd.DataFrame(rows)
avg = r.drop(columns="seed").mean(); avg["seed"] = "mean"
tgt = pd.Series({"seed": "target", "AC_IA": .63, "CM_AA": .59, "IR_RA": .52, "between_mean": .35, "between_min": np.nan, "between_max": np.nan})
out = pd.concat([r, avg.to_frame().T, tgt.to_frame().T]); out.to_csv(os.path.join(RES, "realized_correlations.csv"), index=False)
print(out.round(3).to_string(index=False))
