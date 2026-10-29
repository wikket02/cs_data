"""
Full analysis pipeline for the two-stage Composite Readiness Score praxis.
Run from this folder:  python analysis.py
Every number reported in Chapters 3-5 and Appendix A is read from the files this script writes to ../results.
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from generate import generate_dataset, DOMAINS, WEIGHTS, CLUSTERS, BETWEEN, NOISE_FRAC, SEEDS

PRIMARY_SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

# Domain-specific TEI amplification multipliers (praxis Section 3.5, Table of TEI multipliers)
TEI_MULT_TABLE = {
    "AC": {"low": 1.10, "moderate": 1.32, "high": 1.68},
    "IR": {"low": 1.12, "moderate": 1.35, "high": 1.72},
    "CM": {"low": 1.08, "moderate": 1.22, "high": 1.48},
    "AA": {"low": 1.06, "moderate": 1.19, "high": 1.42},
    "IA": {"low": 1.05, "moderate": 1.16, "high": 1.36},
    "RA": {"low": 1.09, "moderate": 1.24, "high": 1.46},
}

def run_regression(df):
    X = sm.add_constant(df[DOMAINS])
    model = sm.OLS(df["Readiness"], X).fit()
    vif = {d: variance_inflation_factor(X.values, i + 1) for i, d in enumerate(DOMAINS)}
    resid = model.resid
    sw = stats.shapiro(resid)
    bp = het_breuschpagan(resid, X)
    dw = durbin_watson(resid)
    cd = model.get_influence().cooks_distance[0]
    infl = np.where(cd > 4 / len(df))[0]
    m2 = sm.OLS(df["Readiness"].drop(df.index[infl]), X.drop(df.index[infl])).fit()
    chg = ((m2.params[DOMAINS] - model.params[DOMAINS]).abs() / model.params[DOMAINS].abs() * 100)
    cook = dict(n_influential=int(len(infl)), threshold=4 / len(df), max_cooks_d=float(cd.max()),
                max_pct_coef_change=float(chg.max()), pct_coef_change=chg.to_dict(),
                refit_significant=[d for d in DOMAINS if m2.pvalues[d] < .05], refit_p={d: float(m2.pvalues[d]) for d in DOMAINS})
    return model, vif, sw, bp, dw, cook

def descriptive_table(df):
    rows = []
    for d in DOMAINS + ["Readiness"]:
        s = df[d]
        rows.append([d, s.mean(), s.std(ddof=1), s.min(), s.max(), stats.skew(s), stats.kurtosis(s)])
    return pd.DataFrame(rows, columns=["Variable", "Mean", "SD", "Min", "Max", "Skew", "Kurt"])

def zone_report(df):
    s = df["Readiness"]; n = len(s)
    q1, q2, q3 = s.quantile([.25, .5, .75])
    low = int((s < 33.33).sum()); high = int((s > 66.67).sum()); mod = n - low - high
    return dict(Q1=q1, Median=q2, Q3=q3, IQR=q3 - q1, N=n, low_n=low, low_pct=100 * low / n,
                mod_n=mod, mod_pct=100 * mod / n, high_n=high, high_pct=100 * high / n)

def spearman(x, y):
    return stats.spearmanr(x, y)[0]

def h1_weight_perturbation(df, weights=WEIGHTS, pct=0.20):
    base_score = (sum(df[d] * weights[d] for d in DOMAINS) + df["eps"]) / 3.0 * 100.0
    base_rank = base_score.rank()
    rows = []
    for d in DOMAINS:
        for direction, sign in [("+", 1), ("-", -1)]:
            wk = weights[d] * (1 + sign * pct)
            neww = {j: (wk if j == d else weights[j] * (1 - wk) / (1 - weights[d])) for j in DOMAINS}
            score = (sum(df[j] * neww[j] for j in DOMAINS) + df["eps"]) / 3.0 * 100.0
            rows.append(dict(domain=d, direction=direction, nominal_weight=weights[d], adj_weight=round(wk, 4),
                             rho=spearman(base_rank, score.rank()), mean_shift=(score - base_score).abs().mean()))
    return pd.DataFrame(rows)

def priority(df, betas, level):
    means = df[DOMAINS].mean()
    return pd.Series({d: betas[d] * (3.0 - means[d]) * TEI_MULT_TABLE[d][level] for d in DOMAINS})

def h3_tei(df, betas):
    levels = ["low", "moderate", "high"]
    pr = {l: priority(df, betas, l) for l in levels}
    rk = {l: pr[l].rank(ascending=False) for l in levels}
    pairs = [("low", "moderate"), ("low", "high"), ("moderate", "high")]
    return pd.DataFrame([dict(pair=f"{a}-{b}", rho=spearman(rk[a], rk[b])) for a, b in pairs]), pr, rk

def h2_corr_perturbation(df_nom, betas_nom, seed, delta=0.10, noise_frac=NOISE_FRAC, **gen):
    nom_coef_rank = betas_nom[DOMAINS].rank(ascending=False)
    nom_ap_rank = priority(df_nom, betas_nom, "moderate").rank(ascending=False)
    params = {"AC_IA": .63, "CM_AA": .59, "IR_RA": .52, "between": .35}
    rows = []
    for pname, pval in params.items():
        for sign in [1, -1]:
            newval = min(max(pval + sign * delta, -.95), .95)
            clusters = dict(CLUSTERS); between = BETWEEN
            if pname == "AC_IA": clusters[("AC", "IA")] = newval
            elif pname == "CM_AA": clusters[("CM", "AA")] = newval
            elif pname == "IR_RA": clusters[("IR", "RA")] = newval
            else: between = newval
            dfp, _, _ = generate_dataset(seed=seed, clusters=clusters, between=between, noise_frac=noise_frac, **gen)
            mp = sm.OLS(dfp["Readiness"], sm.add_constant(dfp[DOMAINS])).fit()
            bp_ = mp.params.drop("const")
            rows.append(dict(param=pname, direction="+" if sign > 0 else "-", new_value=round(newval, 2),
                             rho_coef=spearman(nom_coef_rank, bp_[DOMAINS].rank(ascending=False)),
                             rho_priority=spearman(nom_ap_rank, priority(dfp, bp_, "moderate").rank(ascending=False))))
    return pd.DataFrame(rows)

def full_seed(seed, noise_frac=NOISE_FRAC, **gen):
    """gen may contain means=, sds=, weights= overrides (used by assumption_sensitivity.py)."""
    df, R, sigma = generate_dataset(seed=seed, noise_frac=noise_frac, **gen)
    model, vif, sw, bp, dw, cook = run_regression(df)
    betas = model.params.drop("const")
    h1 = h1_weight_perturbation(df, **({"weights": gen["weights"]} if "weights" in gen else {}))
    h3, pr, rk = h3_tei(df, betas)
    h2 = h2_corr_perturbation(df, betas, seed, noise_frac=noise_frac, **gen)
    return dict(df=df, model=model, vif=vif, sw=sw, bp=bp, dw=dw, cook=cook, betas=betas, h1=h1, h3=h3, pr=pr, rk=rk, h2=h2, sigma=sigma)

if __name__ == "__main__":
    allres = {s: full_seed(s) for s in SEEDS}
    r = allres[PRIMARY_SEED]; df = r["df"]; m = r["model"]
    # ---- Primary seed outputs ----
    descriptive_table(df).to_csv(f"{RES}/table1_descriptives.csv", index=False)
    corr = df[DOMAINS + ["Readiness"]].corr(); corr.to_csv(f"{RES}/table2_correlations.csv")
    pv = m.pvalues; ci = m.conf_int()
    sd_ratio = df[DOMAINS].std(ddof=1) / df["Readiness"].std(ddof=1)
    reg = pd.DataFrame({"b": m.params, "SE": m.bse, "t": m.tvalues, "p": pv, "CI_low": ci[0], "CI_high": ci[1]})
    reg["beta"] = [np.nan] + [m.params[d] * sd_ratio[d] for d in DOMAINS]
    reg.to_csv(f"{RES}/table3_regression.csv")
    diag = dict(r2=m.rsquared, adj_r2=m.rsquared_adj, F=m.fvalue, F_p=m.f_pvalue, df_model=int(m.df_model), df_resid=int(m.df_resid),
                vif=r["vif"], shapiro_W=r["sw"][0], shapiro_p=r["sw"][1], bp_lm=r["bp"][0], bp_p=r["bp"][1],
                dw=r["dw"], cook=r["cook"], sigma=r["sigma"], zones=zone_report(df),
                theoretical_r2=1 / (1 + NOISE_FRAC ** 2),
                n_clipped_low=int((((df.WeightedScore + df.eps) / 3 * 100) < 0).sum()),
                n_clipped_high=int((((df.WeightedScore + df.eps) / 3 * 100) > 100).sum()))
    json.dump(diag, open(f"{RES}/table4_diagnostics.json", "w"), indent=2, default=float)
    rows = []
    for d in DOMAINS:
        rows.append(dict(domain=d, beta=r["betas"][d], mean_deficiency=3.0 - df[d].mean(),
                         low=r["pr"]["low"][d], moderate=r["pr"]["moderate"][d], high=r["pr"]["high"][d],
                         rank_low=int(r["rk"]["low"][d]), rank_moderate=int(r["rk"]["moderate"][d]), rank_high=int(r["rk"]["high"][d])))
    pd.DataFrame(rows).to_csv(f"{RES}/table5_6_tei_priorities.csv", index=False)
    r["h3"].to_csv(f"{RES}/h3_tei_robustness.csv", index=False)
    r["h1"].to_csv(f"{RES}/table7_weight_perturbation.csv", index=False)
    r["h2"].to_csv(f"{RES}/h2_correlation_perturbation.csv", index=False)
    # ---- Five-seed summary ----
    srows, hrows = [], []
    for s in SEEDS:
        x = allres[s]; mm = x["model"]; b = x["betas"]; dd = x["df"]
        srows.append(dict(seed=s, r2=mm.rsquared, adj_r2=mm.rsquared_adj, F=mm.fvalue, F_p=mm.f_pvalue,
                          **{f"b_{d}": b[d] for d in DOMAINS}, top_domain=b.idxmax(),
                          n_sig=sum(mm.pvalues[d] < .05 for d in DOMAINS),
                          sig_domains=",".join(d for d in DOMAINS if mm.pvalues[d] < .05),
                          readiness_mean=dd["Readiness"].mean(), readiness_sd=dd["Readiness"].std(ddof=1),
                          h1_min_rho=x["h1"]["rho"].min(), h1_pass=bool(x["h1"]["rho"].min() >= .95 - 1e-9),
                          h3_min_rho=x["h3"]["rho"].min(), h3_pass=bool(x["h3"]["rho"].min() >= .80 - 1e-9),
                          shapiro_p=x["sw"][1], bp_p=x["bp"][1], dw=x["dw"], max_vif=max(x["vif"].values()),
                          n_cook=x["cook"]["n_influential"]))
        hrows.append(dict(seed=s, min_rho_coef=x["h2"]["rho_coef"].min(), min_rho_priority=x["h2"]["rho_priority"].min(),
                          h2_pass=bool(min(x["h2"]["rho_coef"].min(), x["h2"]["rho_priority"].min()) >= .80 - 1e-9)))
    pd.DataFrame(srows).to_csv(f"{RES}/five_seed_summary.csv", index=False)
    pd.DataFrame(hrows).to_csv(f"{RES}/h2_five_seed_summary.csv", index=False)
    print("Analysis complete. Results written to", os.path.abspath(RES))
    print(f"Primary seed {PRIMARY_SEED}: Readiness mean = {df.Readiness.mean():.2f}, adj R2 = {m.rsquared_adj:.3f}, F = {m.fvalue:.2f}")
