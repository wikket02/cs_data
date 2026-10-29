"""Compare numeric table values in the manuscript (.docx) with the saved tables.
Usage: python verify_manuscript_values.py path/to/manuscript.docx
Matches tables by header text; reports every mismatching cell. Values are compared after
normalising leading zeros, minus signs and dashes."""
import sys, os, re, pandas as pd, docx
TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
norm = lambda v: re.sub(r"^(-?)0\.", r"\1.", v.replace("−", "-").replace("–", "-").replace("—", "-").replace("**", "").strip())
d = docx.Document(sys.argv[1]); T = d.tables
def rd(f): return pd.read_csv(os.path.join(TAB, f + ".csv"), dtype=str, keep_default_na=False)
def find(h):
    m = [t for t in T if [c.text.strip() for c in t.rows[0].cells][:len(h)] == h]
    assert m, f"table with header {h} not found"; return m[0]
checks = [
 (["Variable", "Mean", "SD", "Min", "Max", "Skew", "Kurt"], "table1_descriptives", [1,2,3,4,5,6], 1),
 (["Predictor", "b", "SE", "t", "p", "95% CI", "Beta"], "table3_regression", [1,2,3,4,5,6], 1),
 (["Test", "Statistic", "Value", "Decision Criterion", "Result"], "table4_diagnostics", [1,2,3,4], 1),
 (["Domain", "Beta", "Mean Deficiency", "Low TEI", "Moderate TEI", "High TEI"], "table5_tei_adjusted_scores", [1,2,3,4,5], 1),
 (["Perturbed Domain", "Direction", "Adjusted Weight", "Nominal Weight"], "table7_weight_sensitivity", [2,3,4,5], 2),
 (["Seed", "Adj. R²", "F(6,193)"], "tableA1_five_seed_summary", list(range(11)), 0),
 (["Seed", "Min ρ (coefficient ordering)"], "tableA2_h2_five_seed", [0,1,2,3], 0),
]
bad = n = 0
for hdr, csv, cols, c0 in checks:
    t = find(hdr); df = rd(csv)
    for i in range(len(df)):
        for k, c in enumerate(cols):
            a = norm(t.rows[1 + i].cells[c0 + k].text); b = norm(df.iloc[i, c]); n += 1
            if a != b: bad += 1; print(f"MISMATCH {csv} row {i+1} col {c}: manuscript '{a}' vs saved '{b}'")
t2 = pd.read_csv(os.path.join(TAB, "table2_correlations.csv"), index_col=0); t = find(["Variable", "AC", "IR"])
for i in range(1, 7):
    for j in range(i):
        v = t2.iloc[i, j]; b = ('-' if v < 0 else '') + f".{round(abs(v)*100):02d}"; a = norm(t.rows[i+1].cells[j+1].text); n += 1
        if a != b: bad += 1; print(f"MISMATCH table2 [{i},{j}]: manuscript '{a}' vs saved '{b}'")

# ---- Tables A3 to A6, generated from the saved results ----
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
f3 = lambda x: f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"
def cmp(tbl, rows, label):
    global n, bad
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            a = norm(tbl.rows[1 + i].cells[j].text); b = norm(str(v)); n += 1
            if a != b: bad += 1; print(f"MISMATCH {label} row {i+1} col {j}: manuscript '{a}' vs saved '{b}'")
ns = pd.read_csv(os.path.join(RES, "noise_sensitivity.csv"))
cmp(find(["Noise fraction", "Theoretical R²"]), [[f"{r.noise_frac:.2f}", f3(r.theoretical_r2), f3(r.mean_observed_r2), f3(r.mean_adj_r2), f"{r.mean_n_sig:.1f}", int(r.top_domain_AC_seeds), int(r.h1_pass_seeds), int(r.h3_pass_seeds), int(r.h2_pass_seeds)] for r in ns.itertuples()], "A3")
asn = pd.read_csv(os.path.join(RES, "assumption_sensitivity.csv"))
cmp(find(["Scenario", "Mean adj. R²"]), [[r.scenario, f3(r.mean_adj_r2), r.top_domain_by_seed, int(r.top_is_AC_seeds), f"{r.mean_n_sig:.1f}", int(r.h1_pass_seeds), int(r.h3_pass_seeds), int(r.h2_pass_seeds)] for r in asn.itertuples()], "A4")
cmp(find(["Condition", "Population ρ (N = 1,000,000)"]), rd("tableA5_h2_decomposition").values.tolist(), "A5")
cmp(find(["Domain", "Parameter mean"]), rd("tableA6_realized_moments").values.tolist(), "A6")
# ---- key numbers quoted in running text must appear in the manuscript ----
text = " ".join(p.text for p in d.paragraphs) + " " + " ".join(c.text for t in T for r in t.rows for c in r.cells)
text = text.replace("\u2212", "-")
h2 = pd.read_csv(os.path.join(RES, "h2_five_seed_summary.csv")); pr = pd.read_csv(os.path.join(RES, "h2_decomposition_pass_rates.csv")).iloc[0]
dec = pd.read_csv(os.path.join(RES, "h2_decomposition.csv")); fs = pd.read_csv(os.path.join(RES, "five_seed_summary.csv"))
pm = lambda x: f"{x:.3f}".lstrip("0")
need = {"H2 pass rate, profiles held fixed": f"{100*pr.h2_pass_rate_actual:.1f}%", "placebo failure rate": f"{dec.iloc[0].pct_either_below_80:.1f}%",
        "lowest H2 seed minimum": pm(min(h2.min_rho_coef.min(), h2.min_rho_priority.min())), "primary-seed H2 minimum": pm(min(h2.iloc[0].min_rho_coef, h2.iloc[0].min_rho_priority)),
        "largest scenario failure rate": f"{dec.iloc[1:].pct_either_below_80.max():.1f}%", "smallest scenario failure rate": f"{dec.iloc[1:].pct_either_below_80.min():.1f}%",
        "primary adj. R2": pm(fs.iloc[0].adj_r2), "primary Readiness mean": f"{fs.iloc[0].readiness_mean:.2f}",
        "Readiness mean range": f"{fs.readiness_mean.min():.2f} to {fs.readiness_mean.max():.2f}"}
for k, v in need.items():
    n += 1
    if v not in text: bad += 1; print(f"MISSING in text: {k} = {v}")
# ---- stale statements that must not reappear ----
for stale in ["four of five datasets", "four of the five independently generated datasets", "passed in only one of the five seeds", "H2 is not supported", "H2 was not supported", "match Section 3.4.2"]:
    n += 1
    if stale in text and "earlier implementation" not in text[max(0, text.find(stale)-400):text.find(stale)+400]:
        bad += 1; print(f"STALE statement present: {stale}")
print(f"{n} checks, {bad} problems"); sys.exit(1 if bad else 0)
