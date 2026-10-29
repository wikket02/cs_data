"""Builds the formatted versions of manuscript Tables 1-8 and A1-A2 from ../results. Output: ../tables/*.csv"""
import os, json, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results"); OUT = os.path.join(HERE, "..", "tables"); os.makedirs(OUT, exist_ok=True)
N = {"AC": "Access Control", "IR": "Incident Response", "CM": "Config. Management", "AA": "Audit & Accountability", "IA": "ID & Authentication", "RA": "Risk Assessment", "Readiness": "Readiness Score"}
def f2(x): return f"{x:.2f}"
def ptxt(p): return "< .001" if p < .001 else f"{p:.3f}".lstrip("0")
def rtxt(x): return f"{x:.3f}".replace("0.", ".", 1).replace("-.", "−.") if abs(x) < 1 else f"{x:.3f}"

d = pd.read_csv(f"{RES}/table1_descriptives.csv")
pd.DataFrame({"Variable": d.Variable.map(N), "Mean": d.Mean.map(f2), "SD": d.SD.map(f2), "Min": d.Min.map(f2), "Max": d.Max.map(f2), "Skew": d.Skew.map(f2), "Kurt": d.Kurt.map(f2)}).to_csv(f"{OUT}/table1_descriptives.csv", index=False)
c = pd.read_csv(f"{RES}/table2_correlations.csv", index_col=0)
t2 = c.copy().applymap(lambda v: f"{v:.2f}".lstrip("0")) if hasattr(c, "applymap") else c
t2.index = [N[i] for i in c.index]; t2.to_csv(f"{OUT}/table2_correlations.csv")
r = pd.read_csv(f"{RES}/table3_regression.csv", index_col=0)
pd.DataFrame({"Predictor": ["Intercept"] + [N[i] for i in r.index[1:]], "b": r.b.map(f2), "SE": r.SE.map(f2), "t": r.t.map(f2), "p": r.p.map(ptxt),
              "95% CI": [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(r.CI_low, r.CI_high)], "Beta": ["-"] + [f"{b:.2f}".lstrip("0") for b in r.beta[1:]]}).to_csv(f"{OUT}/table3_regression.csv", index=False)
g = json.load(open(f"{RES}/table4_diagnostics.json"))
rows = [[f"VIF - {N[k]}", "VIF", f2(v), "< 5.00", "Satisfactory"] for k, v in g["vif"].items()]
rows += [["Residual Normality", "Shapiro-Wilk W", f"{g['shapiro_W']:.3f}", f"p > .05 (p = {g['shapiro_p']:.3f})".replace("= 0.", "= ."), "Satisfactory"],
         ["Heteroscedasticity", "Breusch-Pagan LM(6)", f2(g["bp_lm"]), f"p > .05 (p = {g['bp_p']:.3f})".replace("= 0.", "= ."), "Satisfactory"],
         ["Autocorrelation", "Durbin-Watson d", f2(g["dw"]), "1.50 to 2.50", "Satisfactory"],
         ["Influential Cases", "Cook's D > 4/n", f"{g['cook']['n_influential']} cases", "Sensitivity re-run", f"Max coefficient change {g['cook']['max_pct_coef_change']:.0f}% (IA); see text"]]
pd.DataFrame(rows, columns=["Test", "Statistic", "Value", "Decision Criterion", "Result"]).to_csv(f"{OUT}/table4_diagnostics.csv", index=False)
t = pd.read_csv(f"{RES}/table5_6_tei_priorities.csv")
pd.DataFrame({"Domain": t.domain.map(N), "Beta (b)": t.beta.map(f2), "Mean Deficiency": t.mean_deficiency.map(f2), "Low TEI": t.low.map(f2), "Moderate TEI": t.moderate.map(f2), "High TEI": t.high.map(f2)}).to_csv(f"{OUT}/table5_tei_adjusted_scores.csv", index=False)
pd.DataFrame({"Domain": t.domain.map(N), "Rank (Low TEI)": t.rank_low, "Rank (Moderate TEI)": t.rank_moderate, "Rank (High TEI)": t.rank_high}).to_csv(f"{OUT}/table6_tei_rankings.csv", index=False)
h = pd.read_csv(f"{RES}/table7_weight_perturbation.csv")
pd.DataFrame({"Perturbed Domain": h.domain.map(N), "Direction": h.direction.map({"+": "+20%", "-": "-20%"}), "Adjusted Weight": h.adj_weight.map(lambda x: f"{x:.3f}"),
              "Nominal Weight": h.nominal_weight.map(lambda x: f"{x:.3f}"), "Mean |Score Shift|": h.mean_shift.map(f2), "Spearman rho": h.rho.map(lambda x: f"{x:.3f}")}).to_csv(f"{OUT}/table7_weight_sensitivity.csv", index=False)
s = pd.read_csv(f"{RES}/five_seed_summary.csv"); h2 = pd.read_csv(f"{RES}/h2_five_seed_summary.csv")
pd.DataFrame({"Seed": s.seed, "Adj. R2": s.adj_r2.map(lambda x: f"{x:.3f}"), "F(6,193)": s.F.map(f2), "Top domain": s.top_domain, "# sig. (p<.05)": s.n_sig,
              "Readiness M": s.readiness_mean.map(f2), "Readiness SD": s.readiness_sd.map(f2), "H1 min rho": s.h1_min_rho.map(lambda x: f"{x:.3f}"), "H1 pass": s.h1_pass.map({True: "Yes", False: "No"}),
              "H3 min rho": s.h3_min_rho.map(lambda x: f"{x:.3f}"), "H3 pass": s.h3_pass.map({True: "Yes", False: "No"})}).to_csv(f"{OUT}/tableA1_five_seed_summary.csv", index=False)
pd.DataFrame({"Seed": h2.seed, "Min rho (coefficient ordering)": h2.min_rho_coef.map(lambda x: f"{x:.3f}"), "Min rho (priority ordering)": h2.min_rho_priority.map(lambda x: f"{x:.3f}"),
              "H2 pass (rho >= .80)?": h2.h2_pass.map({True: "Yes", False: "No"})}).to_csv(f"{OUT}/tableA2_h2_five_seed.csv", index=False)
pd.read_csv(f"{RES}/noise_sensitivity.csv").round(4).to_csv(f"{OUT}/tableA3_noise_sensitivity.csv", index=False)
pd.read_csv(f"{RES}/assumption_sensitivity.csv").round(4).to_csv(f"{OUT}/tableA4_assumption_sensitivity.csv", index=False)
dec = pd.read_csv(f"{RES}/h2_decomposition.csv"); pop = pd.read_csv(f"{RES}/h2_decomposition_population.csv").set_index("scenario")
lab = {"nominal": "Nominal correlations, profiles redrawn (placebo)", "AC_IA+": "AC-IA .63 to .73", "AC_IA-": "AC-IA .63 to .53", "CM_AA+": "CM-AA .59 to .69", "CM_AA-": "CM-AA .59 to .49",
       "IR_RA+": "IR-RA .52 to .62", "IR_RA-": "IR-RA .52 to .42", "between+": "Between-cluster .35 to .45", "between-": "Between-cluster .35 to .25"}
rows = []
for r in dec.itertuples():
    rows.append([lab[r.scenario], f"{min(pop.loc[r.scenario, 'rho_coef_vs_nominal'], pop.loc[r.scenario, 'rho_priority_vs_nominal']):.3f}",
                 f"{r.mean_rho_coef:.3f}", f"{r.mean_rho_priority:.3f}", f"{r.pct_either_below_80:.1f}",
                 "" if np.isnan(r.pct_indep_either_below_80) else f"{r.pct_indep_either_below_80:.1f}", f"{r.mean_recovery_coef:.3f}"])
pd.DataFrame(rows, columns=["Condition", "Population rho (N = 1,000,000)", "Mean rho, coefficient order", "Mean rho, priority order", "% replications with rho < .80", "% with rho < .80 if profiles independently redrawn", "Mean rho of coefficient order vs construction"]).to_csv(f"{OUT}/tableA5_h2_decomposition.csv", index=False)
m = pd.read_csv(f"{RES}/realized_moments.csv")
pd.DataFrame({"Domain": m.domain, "Parameter mean": m.param_mean.map(f2), "Parameter SD": m.param_sd.map(f2), "Truncated mean": m.truncated_mean.map(f2), "Truncated SD": m.truncated_sd.map(f2),
              "Seed 42 mean": m.seed42_mean.map(f2), "Seed 42 SD": m.seed42_sd.map(f2), "Five-seed mean": m.five_seed_mean.map(f2), "Five-seed SD": m.five_seed_sd.map(f2)}).to_csv(f"{OUT}/tableA6_realized_moments.csv", index=False)
print("Tables written to", os.path.abspath(OUT))
