# Mas & Pallais (2017) human data

Files from the replication package for Mas & Pallais (2017, AER), "Valuing Alternative Work
Arrangements": [openICPSR 113162](https://doi.org/10.3886/E113162V1). Only the experiment data is
included; the package's UAS, CPS and Stata files are not used.

| file | contents |
|---|---|
| experimentdata.dta | one row per job applicant (7,058): `treatment_number`, `chose_position1`, `wagegap`, `female`, ... |
| treatment_codes.xlsx | which jobs are positions A and B in each treatment |
| readme.docx | the package's readme |

`chose_position1` is 1 if the applicant chose position A (position 1 in the codes is A, not the
position shown first). `wagegap` is **B's wage minus A's**, so the paper's wage gap is `-wagegap`.

src/experiments/mas_pallais uses the five main treatments (1-5; the paper's Table 2) and reproduces Table 5's
means and medians; see src/experiments/mas_pallais/analysis.py.
