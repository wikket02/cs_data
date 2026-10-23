"""Regenerates Figures 3-8 of the praxis from the primary-seed dataset (seed 42). Figures 1-2 are architecture diagrams with no data.
Output: ../figures/figure{3..8}_*.png"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from analysis import full_seed, PRIMARY_SEED, DOMAINS, RES
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "figures"); os.makedirs(OUT, exist_ok=True)
NAVY, MID, LIGHT, ORANGE, GREEN = "#1f3864", "#2e75b6", "#bdd7ee", "#c55a11", "#375623"
NAMES = {"AC": "Access Control (AC)", "IR": "Incident Response (IR)", "CM": "Config. Management (CM)",
         "AA": "Audit & Accountability (AA)", "IA": "ID & Authentication (IA)", "RA": "Risk Assessment (RA)"}
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name, px):
    fig.set_size_inches(px[0] / 200, px[1] / 200); fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=200); plt.close(fig)

r = full_seed(PRIMARY_SEED); df = r["df"]; m = r["model"]; y = df["Readiness"]

# Figure 3: box plots of the six domain scores
fig, ax = plt.subplots()
bp = ax.boxplot([df[d] for d in DOMAINS], patch_artist=True, widths=.55, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=NAVY))
for b in bp["boxes"]: b.set(facecolor=LIGHT, edgecolor=NAVY)
for med in bp["medians"]: med.set(color=ORANGE, linewidth=2)
ax.set_xticklabels([NAMES[d].replace(" (", "\n(") for d in DOMAINS], fontsize=10)
ax.set_ylabel("Domain maturity score (0-3)"); ax.set_ylim(-.1, 3.1)
ax.axhline(1.5, color="grey", ls=":", lw=1)
save(fig, "figure3_domain_boxplots.png", (2371, 1374))

# Figure 4: readiness histogram with zone boundaries
fig, ax = plt.subplots()
n, bins, patches = ax.hist(y, bins=24, edgecolor=NAVY)
for p, left in zip(patches, bins[:-1]):
    c = (left + (bins[1] - bins[0]) / 2)
    p.set_facecolor("#fad9d5" if c < 33.33 else ("#d5f5e3" if c > 66.67 else LIGHT))
xs = np.linspace(y.min() - 5, y.max() + 5, 300)
ax.plot(xs, stats.norm.pdf(xs, y.mean(), y.std(ddof=1)) * len(y) * (bins[1] - bins[0]), color=NAVY, lw=3, label="Normal curve")
ax.axvline(33.33, color=ORANGE, ls="--", lw=2.5, label="Low / Moderate boundary (33.33)")
ax.axvline(66.67, color=GREEN, ls="--", lw=2.5, label="Moderate / High boundary (66.67)")
ax.axvline(y.mean(), color="#5b6b8c", ls=":", lw=2)
z = pd.read_json(os.path.join(RES, "table4_diagnostics.json"))["zones"] if False else None
low, high = int((y < 33.33).sum()), int((y > 66.67).sum()); mod = len(y) - low - high
ax.set_ylim(0, 34)
ax.text(y.mean() + 1, 31.2, f"Mean = {y.mean():.2f}", color=NAVY, fontsize=12)
ax.text(10, 17, f"Low Readiness\nn = {low}  ({100*low/len(y):.1f}%)", ha="center", color="#9c2a1f", bbox=dict(fc="white", ec=ORANGE))
ax.text(50, 27, f"Moderate Readiness\nn = {mod}  ({100*mod/len(y):.1f}%)", ha="center", color=NAVY, bbox=dict(fc="white", ec=MID))
ax.text(82, 14, f"High Readiness\nn = {high}  ({100*high/len(y):.1f}%)", ha="center", color=GREEN, bbox=dict(fc="white", ec=GREEN))
ax.set_xlabel("Composite Readiness Score (0-100)"); ax.set_ylabel("Frequency (Number of Supplier Profiles)"); ax.legend(loc="upper right", fontsize=9)
save(fig, "figure4_readiness_distribution.png", (2171, 1271))

# Figure 5: standardized coefficients with significance
reg = pd.read_csv(os.path.join(RES, "table3_regression.csv"), index_col=0)
order = reg.loc[DOMAINS].sort_values("beta", ascending=True)
fig, ax = plt.subplots()
cols = [NAVY if p < .05 else LIGHT for p in order["p"]]
ax.barh([NAMES[d] for d in order.index], order["beta"], color=cols, edgecolor=NAVY, height=.6)
for i, (d, row) in enumerate(order.iterrows()):
    ptxt = "p < .001" if row["p"] < .001 else f"p = {row['p']:.3f}".replace("0.", ".")
    ax.text(row["beta"] + .006, i, f"β = {row['beta']:.2f}  ({ptxt})", va="center", color=NAVY, fontsize=11)
ax.set_xlim(0, .45); ax.set_xlabel("Standardized Regression Coefficient (β)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=NAVY, ec=NAVY, label="Significant at α = .05"), Patch(fc=LIGHT, ec=NAVY, label="Not significant at α = .05")], loc="lower right", fontsize=10)
save(fig, "figure5_standardized_coefficients.png", (2174, 1271))

# Figure 6: standardized residuals vs fitted
infl = m.get_influence(); sr = infl.resid_studentized_internal; fit = m.fittedvalues
bpl, bpp = r["bp"][0], r["bp"][1]
fig, ax = plt.subplots()
ax.scatter(fit, sr, s=28, facecolors="none", edgecolors=MID, alpha=.8)
ax.axhline(0, color=ORANGE, ls="--", lw=1.5, label="Zero reference line (y = 0)")
lo = sm.nonparametric.lowess(sr, fit, frac=.6); ax.plot(lo[:, 0], lo[:, 1], color=NAVY, lw=2, label="Smoothed trend (LOWESS)")
for v in (-2, 2): ax.axhline(v, color="lightgrey", ls=":", lw=1)
ax.set_xlabel("Fitted Values (Predicted Composite Readiness Score)"); ax.set_ylabel("Standardized Residuals")
ax.set_title(f"Breusch-Pagan LM(6) = {bpl:.2f}, p = {bpp:.3f}", fontsize=11, loc="left"); ax.legend(fontsize=10, loc="upper right")
save(fig, "figure6_residuals_vs_fitted.png", (1971, 1371))

# Figure 7: normal Q-Q plot
fig, ax = plt.subplots()
(osm, osr), (slope, icpt, _) = stats.probplot(sr, dist="norm")
ax.scatter(osm, osr, s=28, facecolors="none", edgecolors=MID, label="Observed quantiles")
ax.plot(osm, slope * osm + icpt, color=ORANGE, ls="--", label="Theoretical normal line")
ax.set_xlabel("Theoretical Quantiles (Standard Normal Distribution)"); ax.set_ylabel("Sample Quantiles (Standardized Residuals)")
ax.text(.97, .05, f"Shapiro-Wilk W = {r['sw'][0]:.3f}, p = {r['sw'][1]:.3f}", transform=ax.transAxes, ha="right", fontsize=11, bbox=dict(fc="white", ec=MID))
ax.legend(fontsize=10, loc="upper left")
save(fig, "figure7_qq_plot.png", (1770, 1471))

# Figure 8: adjusted priority scores by TEI level
fig, ax = plt.subplots()
w = .26; xs = np.arange(len(DOMAINS))
order8 = sorted(DOMAINS, key=lambda d: -r["pr"]["moderate"][d])
for k, (lvl, col, lab) in enumerate([("low", LIGHT, "Low exposure"), ("moderate", MID, "Moderate exposure"), ("high", NAVY, "High exposure")]):
    vals = [r["pr"][lvl][d] for d in order8]
    ax.bar(xs + (k - 1) * w, vals, w, color=col, edgecolor=NAVY, label=lab)
    for x, v in zip(xs + (k - 1) * w, vals): ax.text(x, v + .3, f"{v:.1f}", ha="center", fontsize=8)
for i, d in enumerate(order8): ax.text(i, ax.get_ylim()[1] * .97, f"Rank #{int(r['rk']['moderate'][d])}", ha="center", color=ORANGE, fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([NAMES[d].replace(" (", "\n(") for d in order8], fontsize=10)
ax.set_ylabel("Adjusted Priority Score"); ax.legend(fontsize=10)
save(fig, "figure8_adjusted_priority.png", (2571, 1473))
print("Figures written to", os.path.abspath(OUT))
