# Near-lossless knowledge distillation: a theoretical treatment

`main.pdf` (built from `main.tex` and `sections/`) is a self-contained, proof-oriented account of
distilling an autoregressive teacher $p_{\mathsf T}(y\mid x)$ into a smaller student $q_\theta(y\mid x)$.
Every result is labelled Theorem / Proposition / Lemma / Corollary / Conjecture / Heuristic, and
carries a status tag: *proved*, *proved under stated assumptions*, *proved within a stylised model*,
*known result (sketch)*, and, where applicable, *numerically checked*.

| Task in the brief | Section |
|---|---|
| 1. Formalise "lossless" | §1: TV as worst-case task regret; Pinsker/BH/Hellinger; $\mathrm{JSD}_\beta$ sandwich; local $\chi^2$ equivalence; coverage vs precision event bounds; chain rules pairing each KL direction with its prefix law |
| 2. Decompose the error | §2: master identity (approximation + estimation + optimisation + prefix shift + covariate shift); softmax-bottleneck floor; horizon-free MLE rate; Rao–Blackwell view of soft labels; $O(\varepsilon T)$ / $O(\varepsilon T^2)$ / $O(u\,T\varepsilon)$ compounding theorems; recoverability: $u$ is sharp, can be replaced by its average over the student's errors (not the teacher's prefixes), and is $O(1)$ when a cost-sufficient latent chain of the teacher mixes ($u\le1+2m_{1/2}$) |
| 3. Lower bounds | §3: information-theoretic capacity floor $\ge (I(\mathbf T;Y)-R\ln 2)/(mN)$; one-shot rate–distortion converse; power law $D(R)\asymp R^{-(\alpha-1)}$ and exponential phase; scaling-law predictions |
| 4. Objective design | §4: logit gradients; mode covering vs seeking; temperature and $\tau^2$; Fisher view of dark knowledge; unbiased Rao–Blackwellised estimators; **PA-JSD**, a sequence-level prefix-adaptive $\mathrm{JSD}_\beta$ objective with proven properties |
| 5. Beyond output matching | §5: CCA/CKA; last-layer redundancy; when matching helps (noise, conditioning, compositional hardness); over-constraint theorem; Fisher-aligned matching |
| 6. Synthesis | §6: end-to-end sandwich theorem; status table; three experiments with exact quantities to measure. §7 lists open problems |

## Build

```bash
latexmk -pdf main.tex        # needs a TeX Live with amsmath, natbib, cleveref, lmodern
```

## Numerical checks

```bash
pip install numpy scipy
python3 verification/check_divergences.py   # single-distribution inequalities, gradients, temperature limits
python3 verification/check_sequences.py     # exact enumeration: chain rules, compounding, unbiased estimators
python3 verification/check_capacity.py      # water-filling exponents, softmax-bottleneck floor, K-facts floor
python3 verification/check_representations.py  # over-constraint theorem, Fisher-aligned matching, CCA/CKA facts
python3 verification/check_recoverability.py   # recoverability: sharpness, trap, mixing bounds, value martingale
```

Each script prints `PASS`/`FAIL` per claim. The claims are identified by their LaTeX labels.

## Caveats

References were compiled from memory. The text flags each item whose details (exact parametrisations,
author lists, venues) should be verified before citing externally.
