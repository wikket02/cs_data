"""
Synthetic supplier-profile generator for the Composite Readiness Score model.
Implements the NORTA (NORmal To Anything) method (Cario & Nelson, 1997;
Ghosh & Henderson, 2003) to draw correlated, truncated-normal domain
maturity scores, assigns Threat Exposure Index categories, and adds the
idiosyncratic noise term described in Section 3.4.1 of the praxis.
"""
import numpy as np
from scipy import stats
import pandas as pd

DOMAINS = ["AC","IR","CM","AA","IA","RA"]
MEANS   = {"AC":1.38, "IR":1.09, "CM":1.28, "AA":1.22, "IA":1.35, "RA":1.02}
SDS     = {"AC":0.58, "IR":0.68, "CM":0.61, "AA":0.60, "IA":0.54, "RA":0.69}
WEIGHTS = {"AC":0.25, "IR":0.20, "CM":0.18, "AA":0.15, "IA":0.12, "RA":0.10}
LO, HI = 0.0, 3.0

CLUSTERS = {("AC","IA"):0.63, ("CM","AA"):0.59, ("IR","RA"):0.52}
BETWEEN = 0.35

def build_corr_matrix(clusters=CLUSTERS, between=BETWEEN):
    n = len(DOMAINS)
    R = np.full((n,n), between)
    np.fill_diagonal(R, 1.0)
    idx = {d:i for i,d in enumerate(DOMAINS)}
    for (a,b), val in clusters.items():
        i,j = idx[a], idx[b]
        R[i,j] = R[j,i] = val
    return R

# Platform-independent factorization of the correlation matrix.
# numpy's Generator.multivariate_normal factors the matrix with an SVD whose singular-vector signs depend on the
# LAPACK build, so the same seed could give different datasets (especially for perturbed matrices) on different
# machines, and a perturbed matrix could produce profiles unrelated to the nominal ones. Here:
#  * the nominal matrix uses eigenvectors with a fixed sign convention (the first component with the largest
#    absolute value is positive, multiplied by S0, the pattern that reproduces the datasets shipped with the praxis);
#  * every other matrix aligns each eigenvector's sign with the nominal one, so that a small change in a correlation
#    gives a small change in the simulated profiles for the same seed (common random numbers).
S0 = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0])

def _eig_rows(R):
    w, V = np.linalg.eigh(R)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order].T          # rows are unit eigenvectors, descending eigenvalues

_NOM_ROWS = None
def _nominal_rows():
    global _NOM_ROWS
    if _NOM_ROWS is None:
        _, rows = _eig_rows(build_corr_matrix())
        out = []
        for v in rows:
            k = int(np.where(np.abs(v) > np.abs(v).max() - 1e-9)[0][0])
            out.append(v * np.sign(v[k]))
        _NOM_ROWS = np.array(out) * S0[:, None]
    return _NOM_ROWS

def corr_factor(R):
    """Factor F with F.T @ F = R. Eigenvectors are matched to the nominal matrix's eigenvectors by largest overlap
    (so an eigenvalue crossing cannot pair the wrong directions) and signed to agree with them."""
    from scipy.optimize import linear_sum_assignment
    w, rows = _eig_rows(R)
    nom = _nominal_rows()
    overlap = np.abs(rows @ nom.T)                     # perturbed vector i versus nominal vector j
    ri, cj = linear_sum_assignment(-overlap)
    perm = np.empty(len(w), dtype=int); perm[cj] = ri  # nominal position j takes perturbed vector perm[j]
    w, rows = w[perm], rows[perm]
    sign = np.sign(np.sum(rows * nom, axis=1)); sign[sign == 0] = 1.0
    return np.sqrt(w)[:, None] * (rows * sign[:, None])

def truncnorm_params(mean, sd, lo=LO, hi=HI):
    a = (lo - mean)/sd
    b = (hi - mean)/sd
    return a, b

def generate_dataset(seed, n=200, clusters=CLUSTERS, between=BETWEEN,
                      means=MEANS, sds=SDS, weights=WEIGHTS, noise_frac=0.58):
    rng = np.random.default_rng(seed)
    R = build_corr_matrix(clusters, between)
    Z = rng.standard_normal((n, len(DOMAINS))) @ corr_factor(R)
    U = stats.norm.cdf(Z)
    domain_scores = {}
    for j, d in enumerate(DOMAINS):
        a, b = truncnorm_params(means[d], sds[d])
        domain_scores[d] = stats.truncnorm.ppf(U[:, j], a, b, loc=means[d], scale=sds[d])
    df = pd.DataFrame(domain_scores)

    tei_draw = rng.random(n)
    tei_level = np.where(tei_draw < 0.40, "low", np.where(tei_draw < 0.80, "moderate", "high"))
    tei_mult = np.where(tei_level=="low", 1.10, np.where(tei_level=="moderate", 1.25, 1.50))
    df["TEI_level"] = tei_level
    df["TEI_mult"] = tei_mult

    weighted = sum(df[d]*weights[d] for d in DOMAINS)
    sigma = noise_frac * weighted.std(ddof=1)
    eps = rng.normal(0, sigma, n)
    df["WeightedScore"] = weighted
    df["eps"] = eps
    raw_readiness = (weighted + eps)/3.00*100.0
    df["Readiness"] = raw_readiness.clip(lower=0.0, upper=100.0)
    return df, R, sigma

NOISE_FRAC = 0.58
SEEDS = [42, 101, 202, 303, 404]

if __name__ == "__main__":
    # Regenerate the five datasets shipped in ../data (seeds 42, 101, 202, 303, 404)
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    for s in SEEDS:
        df, R, sigma = generate_dataset(seed=s, noise_frac=NOISE_FRAC)
        df.to_csv(os.path.join(out, f"synthetic_supplier_profiles_seed{s}.csv"), index=False)
        print(f"seed {s}: wrote dataset, sigma = {sigma:.4f}")
