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
print(f"{n} cells compared, {bad} mismatches"); sys.exit(1 if bad else 0)
