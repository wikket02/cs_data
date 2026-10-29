"""Regenerate everything: datasets, results, tables, figures. Run from the code/ folder: python run_all.py"""
import subprocess, sys, os
here = os.path.dirname(os.path.abspath(__file__))
for script in ["generate.py", "analysis.py", "noise_sensitivity.py", "assumption_sensitivity.py",
               "check_realized_correlations.py", "check_realized_moments.py", "h2_decomposition.py", "make_tables.py", "make_figures.py"]:
    print("==> running", script, flush=True)
    subprocess.run([sys.executable, os.path.join(here, script)], check=True, cwd=here)
print("Done. Check results/, tables/, figures/. Then: python verify_manuscript_values.py <manuscript.docx>")
