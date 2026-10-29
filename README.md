# Replication package: Two-Stage Composite Readiness Score Framework for SME Defense Suppliers

Repository: [https://github.com/wikket02/cs_data]

All data in this package are **synthetic**. They were generated from researcher-defined assumptions (distribution parameters,
domain weights, correlation values, noise level) and do not describe any real supplier, assessment, or restricted data.
The parameters are not calibrated to GAO or any other empirical source; no published source located for this study reports
domain-level maturity distributions for SME defense suppliers. NORTA specifies how the chosen marginals and correlations become
a dataset; it does not validate the chosen values.

## Layout
- `code/` generation, analysis, sensitivity, table, figure, and verification scripts
- `data/` five generated datasets (seeds 42, 101, 202, 303, 404; 200 profiles each)
- `results/` saved numerical outputs used in Chapters 3 and 4 and Appendix A
- `tables/` CSV versions of Tables 1 to 7 and A1 to A4
- `figures/` Figures 3 to 8 at the manuscript's pixel dimensions (Figures 1 and 2 are data-free diagrams and are not included)
- `requirements.txt` dependency versions used (Python 3.13.16)

## Reproduce
```
pip install -r requirements.txt
cd code
python run_all.py                                   # about 25 seconds
python verify_manuscript_values.py ../path/to/manuscript.docx
```
`run_all.py` runs: `generate.py` (datasets), `analysis.py` (Readiness Score, deficiency, regression, diagnostics, H1 to H3,
five-seed summaries), `noise_sensitivity.py` (Table A3), `assumption_sensitivity.py` (Table A4),
`check_realized_correlations.py`, `make_tables.py`, `make_figures.py`.
A clean re-run was tested: the regenerated datasets and results files are identical to those shipped here.

## Model summary
Readiness = clip((sum_k w_k * D_k + e) / 3 * 100, 0, 100), with weights AC .25, IR .20, CM .18, AA .15, IA .12, RA .10;
e ~ N(0, (0.58 * SD(weighted score))^2). Domain scores are truncated normals on [0, 3] correlated through a Gaussian copula
(NORTA). Threat Exposure Index levels (40/40/20 percent at 1.10/1.25/1.50) apply to Stage 2 priorities only.

## Hypotheses (pre-specified thresholds)
H1 weight perturbation +/-20 percent, Spearman rho >= .95 (supported); H2 correlation perturbation +/-.10, rho >= .80 for every
scenario (supported in all five datasets when the random draws are held fixed, with the qualification that the estimated orderings
change under resampling of the profiles even when no correlation is changed; see `h2_decomposition.py`); H3 threat-exposure
scenarios, rho >= .80 (supported).

## Cross-platform reproducibility
`generate.py` factors the correlation matrix with a fixed eigenvector sign convention instead of numpy's SVD-based
`multivariate_normal`, whose singular-vector signs depend on the LAPACK build. Perturbed-correlation datasets reuse the same
random draws as the nominal dataset. Pass/fail thresholds are applied with a 1e-9 tolerance so that a rank correlation of
exactly .80 or .95 cannot flip with floating-point noise. All reported values were produced on Linux with the pinned versions in
`requirements.txt`; if a run on another platform differs, compare it with `results/` and report the file and value.

## Known limits
Figures 1 and 2 source files are not part of this package. The regression recovers the weights used to build the score, so
its coefficients show internal consistency only (see Appendix A, Table A4).
