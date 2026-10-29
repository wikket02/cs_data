"""Separates two things that the pre-specified H2 test mixes together.
When a correlation parameter is changed and the data are regenerated, (a) the correlation structure changes and
(b) the simulated supplier profiles change, so the re-estimated orderings can move because of either. This script
isolates them:
  1. Population level (N = 1,000,000 per scenario): sampling variation removed, only the correlation effect remains.
  2. Placebo H2: the nominal correlations are kept and only the random profiles are redrawn (no correlation change).
  3. Actual H2 (as pre-specified): correlations perturbed, same seed, re-estimated at N = 200.
  4. Recovery: how well each scenario's N = 200 orderings recover the ordering implied by the construction
     (coefficients proportional to the weights; priorities from the construction-implied values).
Writes ../results/h2_decomposition.csv and ../results/h2_decomposition_population.csv"""
import os, numpy as np, pandas as pd
from scipy import stats
from analysis import spearman, TEI_MULT_TABLE, RES
from generate import generate_dataset, DOMAINS, WEIGHTS, CLUSTERS, BETWEEN, MEANS, SDS, LO, HI, NOISE_FRAC, truncnorm_params

K = 1000                 # replications per condition
LEVEL = "moderate"
TOL = 1e-9
SCEN = [("nominal", None, 0)]
for p in ["AC_IA", "CM_AA", "IR_RA", "between"]:
    for s in (1, -1):
        SCEN.append((f"{p}{'+' if s > 0 else '-'}", p, s))

def scen_args(p, s, delta=0.10):
    cl = dict(CLUSTERS); b = BETWEEN
    if p is None: return cl, b
    base = {"AC_IA": .63, "CM_AA": .59, "IR_RA": .52, "between": .35}[p] + s * delta
    if p == "AC_IA": cl[("AC", "IA")] = base
    elif p == "CM_AA": cl[("CM", "AA")] = base
    elif p == "IR_RA": cl[("IR", "RA")] = base
    else: b = base
    return cl, b

def orderings(df):
    X = np.column_stack([np.ones(len(df)), df[DOMAINS].values])
    beta = np.linalg.lstsq(X, df["Readiness"].values, rcond=None)[0][1:]
    b = pd.Series(beta, index=DOMAINS)
    means = df[DOMAINS].mean()
    pr = pd.Series({d: b[d] * (3.0 - means[d]) * TEI_MULT_TABLE[d][LEVEL] for d in DOMAINS})
    return b, pr

def ranks(s): return s.rank(ascending=False)

# construction-implied ("true") values
w = pd.Series(WEIGHTS)
true_beta = w * 100 / 3
tmean = pd.Series({d: stats.truncnorm.mean(*truncnorm_params(MEANS[d], SDS[d]), loc=MEANS[d], scale=SDS[d]) for d in DOMAINS})
true_pr = pd.Series({d: true_beta[d] * (3.0 - tmean[d]) * TEI_MULT_TABLE[d][LEVEL] for d in DOMAINS})

# ---------------- 1. population level ----------------
pop = {}
for name, p, s in SCEN:
    cl, b = scen_args(p, s)
    df, _, _ = generate_dataset(seed=777, n=1_000_000, clusters=cl, between=b)
    pop[name] = orderings(df)
rows = []
for name, p, s in SCEN:
    rows.append(dict(scenario=name, rho_coef_vs_nominal=spearman(ranks(pop["nominal"][0]), ranks(pop[name][0])),
                     rho_priority_vs_nominal=spearman(ranks(pop["nominal"][1]), ranks(pop[name][1])),
                     **{f"b_{d}": pop[name][0][d] for d in DOMAINS}))
pd.DataFrame(rows).to_csv(os.path.join(RES, "h2_decomposition_population.csv"), index=False)

# ---------------- 2-4. N = 200 replications ----------------
nom_cl, nom_b = scen_args(None, 0)
base = {}      # nominal dataset orderings for seeds 1..K
for k in range(K):
    base[k] = orderings(generate_dataset(seed=100000 + k, n=200)[0])
out = []
for name, p, s in SCEN:
    cl, b = scen_args(p, s)
    rc, rp, ac, ap, rec_c, rec_p = [], [], [], [], [], []
    for k in range(K):
        if p is None:
            # placebo: same correlations, independent redraw of profiles
            bb, pp = orderings(generate_dataset(seed=900000 + k, n=200)[0])
        else:
            bb, pp = orderings(generate_dataset(seed=100000 + k, n=200, clusters=cl, between=b)[0])
        rc.append(spearman(ranks(base[k][0]), ranks(bb)));  rp.append(spearman(ranks(base[k][1]), ranks(pp)))
        rec_c.append(spearman(ranks(true_beta), ranks(bb))); rec_p.append(spearman(ranks(true_pr), ranks(pp)))
        if p is not None:
            # same perturbed correlations, independent redraw (full independence) vs nominal at seed k
            b2, p2 = orderings(generate_dataset(seed=500000 + k, n=200, clusters=cl, between=b)[0])
            ac.append(spearman(ranks(base[k][0]), ranks(b2))); ap.append(spearman(ranks(base[k][1]), ranks(p2)))
    rc, rp = np.array(rc), np.array(rp)
    out.append(dict(scenario=name, kind="placebo (redraw only)" if p is None else "perturbed, same seed",
                    mean_rho_coef=rc.mean(), mean_rho_priority=rp.mean(),
                    pct_coef_below_80=100 * np.mean(rc < .80 - TOL), pct_priority_below_80=100 * np.mean(rp < .80 - TOL),
                    pct_either_below_80=100 * np.mean(np.minimum(rc, rp) < .80 - TOL),
                    mean_recovery_coef=np.mean(rec_c), mean_recovery_priority=np.mean(rec_p),
                    pct_indep_either_below_80=(100 * np.mean(np.minimum(ac, ap) < .80 - TOL) if p is not None else np.nan)))
    print(out[-1]["scenario"], round(out[-1]["pct_either_below_80"], 1), flush=True)
res = pd.DataFrame(out); res.to_csv(os.path.join(RES, "h2_decomposition.csv"), index=False)

# ---------------- whole-test pass rates (all 8 scenarios must pass) ----------------
def run_all8(kind):
    passes = 0
    for k in range(K):
        ok = True
        for name, p, s in SCEN[1:]:
            cl, b = scen_args(p, s)
            if kind == "actual":
                bb, pp = orderings(generate_dataset(seed=100000 + k, n=200, clusters=cl, between=b)[0])
            else:   # placebo: nominal correlations, independent redraw for each of the 8 comparisons
                bb, pp = orderings(generate_dataset(seed=1_000_000 + 8 * k + [x[0] for x in SCEN[1:]].index(name), n=200)[0])
            r = min(spearman(ranks(base[k][0]), ranks(bb)), spearman(ranks(base[k][1]), ranks(pp)))
            if r < .80 - TOL: ok = False
        passes += ok
    return passes / K
summary = dict(K=K, h2_pass_rate_actual=run_all8("actual"), h2_pass_rate_placebo=run_all8("placebo"))
pd.DataFrame([summary]).to_csv(os.path.join(RES, "h2_decomposition_pass_rates.csv"), index=False)
print(summary)
