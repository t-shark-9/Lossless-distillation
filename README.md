# Lossless distillation: the functional information of trained teachers

This repository works on the open problem:

> Characterise, or give computable bounds on, $I_\infty(\pi,D)$ from Theorem 3.1 for realistic priors, such as the distribution of networks produced by a training procedure. Is there a data-dependent estimator of the functional description length of a single trained teacher, such as the minimal $R$ at which some $R$-bit student is $\varepsilon$-lossless, with matching upper and lower bounds?

**Write-up:** [`notes/functional_information.md`](notes/functional_information.md), with statements, proofs, the estimator, and experiments.

## Results in one paragraph

$I_\infty(\pi,D)$ is the entropy of the teacher's function modulo $D$-a.e. equality, and it equals $\lim_{\varepsilon\to0}R_\pi(\varepsilon)$. For realistic priors it takes one of three values:
* $\infty$, under unshared continuous randomness;
* the checkpoint entropy minus the log-size of the permutation/sign symmetry orbit, in finite precision;
* at most the seed entropy, when training is bit-reproducible and the decoder knows the recipe.

So exact lossless distillation is either impossible or trivial. The informative quantity is the rate–distortion function $R_\pi(\varepsilon)$ in function space. It is bounded above by Kolmogorov's reverse water-filling formula over the $L^2(D)$ spectrum of the retrained-teacher ensemble; this bound is exact for lazy/NTK training. It is bounded below by seed/branch conditioning and a projected Shannon bound. Its slope is a scale-dependent functional dimension.

For a single teacher, the $\varepsilon$-tilted functional information $\jmath_\pi(f,\varepsilon)$ is its description length up to $\tfrac12\log_2 k+O(1)$ bits:
* no language beats it on more than a $2^{-\gamma}$ fraction of teachers;
* a prior-matched language attains it for every teacher.

Two impossibility theorems show why the lower bound must depend on a model. The prior-free FDL has no nontrivial computable lower bounds. From $N$ black-box retrains, no procedure can certify more than about $2\log_2N$ bits.

## Code (`code/`, run from inside `code/`)

| script | what it does |
|---|---|
| `fdl.py` | closed forms for Gaussian priors: water-filling rate, tilted information, dispersion, achievability gap, ball masses |
| `verify_gaussian.py` | numerical checks of $\mathbb E\jmath=R$, $\mathrm{Var}\,\jmath=V$, the Csiszár condition, the tilted identity, and $\Delta_f\approx\frac12\log_2(\pi k_a)$ |
| `verify_converse.py` | fixed-length converse vs random-coding achievability; entropy-coded scalar quantiser |
| `mlp_prior.py` | trains an ensemble of MLP teachers (numpy, vectorised) under a seed-only or seed+data prior |
| `analyze_mlp_prior.py` | spectra, Jacobian rank, collision-entropy certified bound, quantised-teacher code |
| `analyze_mlp_crossfit.py` | cross-fitted estimator of $R_\pi(\varepsilon)$ and $\jmath(f,\varepsilon)$; test against real codebooks of retrained teachers |
| `make_figures.py` | figures in `results/` |

```bash
pip install -r requirements.txt
cd code
python3 verify_gaussian.py && python3 verify_converse.py
python3 mlp_prior.py && python3 mlp_prior.py --resample_data --out ../results/mlp_seed_data.npz
python3 analyze_mlp_prior.py && python3 analyze_mlp_crossfit.py && python3 make_figures.py
```

The teacher ensembles (`results/*.npz`, about 10 MB each) are not committed. `mlp_prior.py` regenerates each one in about 90 s on a CPU. The derived numbers (`results/*.json`) and figures are committed.

The source paper containing Theorem 3.1 is not in this repository. The notes reconstruct its setting explicitly and state every result in self-contained terms; see the note at the top of the write-up.
