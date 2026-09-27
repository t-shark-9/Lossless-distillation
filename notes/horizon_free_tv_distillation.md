# Horizon-free, constant-factor distillation with a tractable estimator

**Open problem.** PA-JSD is horizon-free at the population level (Theorem 6.1(iii)), but its gradient estimator has sequence-level score-function variance. Is there an objective that is simultaneously

- **(a)** an exact sequence-level divergence,
- **(b)** agnostically TV-competitive with a constant (not square-root) factor, and
- **(c)** has a gradient estimator whose variance is bounded uniformly in $T$?

For deterministic experts, Foster et al. (2024) suggest that log-loss BC is already horizon-free. The stochastic, misspecified case is open.

> The paper that states this problem is not in this repository. The argument below is self-contained. About PA-JSD it assumes only two things: it is a JSD-type objective that is smooth at $q=p$, and it satisfies $\text{PA-JSD}\le c_0\,\mathrm{TV}$, which is the inequality a horizon-free square-root guarantee needs.

---

## Summary: yes, and the kink is what makes it work

Define the **clipped likelihood-ratio divergences**

$$
\mathcal D_\beta(p\,\|\,q)\;=\;\mathbb E_{x\sim p}\Big[\big(1-(q(x)/p(x))^{\beta}\big)_+\Big]
\;=\;1-\mathbb E_{x\sim p}\big[\min\{1,(q(x)/p(x))^{\beta}\}\big],\qquad \beta\in(0,1].
$$

$\mathcal D_1=\mathrm{TV}$. For any stochastic teacher and any misspecified student class:

| | requirement | what holds for $\mathcal D_\beta$ |
|---|---|---|
| (a) | exact sequence-level divergence | $\mathcal D_\beta=D_{f_\beta}$ on $V^T$ with $f_\beta(t)=(t-t^{1-\beta})\mathbf 1\{t\ge1\}$; $\mathcal D_\beta(p\|q)=0\iff p=q$ |
| (b) | constant-factor agnostic TV | $\beta\,\mathrm{TV}\le\mathcal D_\beta\le\mathrm{TV}$, so $\mathrm{TV}(p,\hat q)\le\mathrm{OPT}/\beta$. From $n$ teacher samples: $\mathrm{TV}(p,\hat q_n)\le\big(\mathrm{OPT}+2\sqrt{\log(2\lvert\mathcal Q\rvert/\delta)/2n}\big)/\beta$, with no dependence on $T$ or $\lvert V\rvert$ |
| (c) | $T$-uniform gradient noise | the gradient is the log-loss (SFT) gradient on teacher samples times a scalar in $[0,\beta]$. It is pointwise dominated by the log-loss BC estimator; its variance is $\le\beta^2\,\mathrm{tr}\,I_T(\theta)$ (for $\beta\ge\tfrac12$); its per-token signal has variance $\le\beta^2/4$ |

Two further results show why this is essentially the only option and what it costs:

1. **The kink is necessary (Theorem 1).** Minimising an $f$-divergence is $C$-competitive in TV, uniformly over $(p,\mathcal Q,T)$, **iff** $f$ has a kink at $1$ and grows linearly. The sharp constant is
   $$C^\star_f=\frac{f(0)+f'(\infty)}{f'(1^+)-f'(1^-)} .$$
   Every divergence that is smooth at $p=q$ is limited to the square-root law: KL, reverse KL, JSD, Hellinger, the generalized JSD family, and PA-JSD under the assumption above. For JSD the law is exact: $\mathrm{TV}(p,\hat q)\le\sqrt{2\ln 2\cdot\mathrm{OPT}}$ always, and this is attained as $\mathrm{OPT}\to0$. So the square root in Theorem 6.1(iii) is a property of any smooth objective, not of the proof technique. The fix is also general: adding $\lambda\,\mathrm{TV}$ to any $D_0\le c_0\mathrm{TV}$ gives $C=1+c_0/\lambda$.
2. **What is not free (Propositions 4 and 5).**
   - An on-policy score-function estimator can have horizon-uniform per-token credit **iff** its sequence reward has $O(1)$ variance. This fails for log-ratio rewards such as reverse KL, and no baseline fixes it.
   - A constant-factor objective is necessarily almost flat far from the teacher: its range is $\le C-1+\eta$ wherever $\mathrm{TV}\ge 1-\eta$.
   - (b) is a statement about global minimisers of an objective that Theorem 1 forces to be non-smooth. A low-variance gradient does not make the optimisation easy.

---

## 1. Setting and formalisation

- **Sequences and teacher.** Sequences are $x\in\mathcal X=V^T$. The teacher $p$ is an arbitrary distribution on $\mathcal X$: stochastic, with no realizability assumed. Access is white-box, as in logit distillation: we can sample $x\sim p$ and evaluate $\log p(x)=\sum_t\log p(x_t\mid x_{<t})$.
- **Students.** The class is $\mathcal Q=\{q_\theta\}$, made of autoregressive models. Every distribution on $V^T$ is autoregressive, so any finite class is admissible. We write $\mathrm{TV}(p,q)=\sum_x(p(x)-q(x))_+=\mathbb E_p[(1-q/p)_+]$ and $\mathrm{OPT}=\inf_{q\in\mathcal Q}\mathrm{TV}(p,q)$.
- **Scores.** The per-token score is $s_t(x)=\nabla_\theta\log q_\theta(x_t\mid x_{<t})$ and the sequence score is $S_\theta(x)=\sum_{t\le T}s_t(x)$. The Fisher information satisfies
  $$\mathrm{tr}\,I_T(\theta)=\mathbb E_{q_\theta}\|S_\theta\|^2=\sum_t\mathbb E_{q_\theta}\|s_t\|^2\le TG^2 \quad\text{when }\|s_t\|\le G.$$
  The cross terms vanish because $\mathbb E_{q_\theta}[s_t\mid x_{<t}]=0$.
- **$f$-divergences.** $D_f(p\|q)=\sum_x q(x)f(p(x)/q(x))$ for convex $f$ with $f(1)=0$. Atoms with $q=0<p$ contribute $p\,f'(\infty)$, where $f'(\infty):=\lim_{t\to\infty}f(t)/t$ and $f(0):=\lim_{t\downarrow0}f(t)$. Two constants matter:
  - $\kappa_f=f'(1^+)-f'(1^-)\ge0$, the size of the kink at $1$;
  - $\Delta_f=f(0)+f'(\infty)$, the value of $D_f$ on two mutually singular laws.

**Formalising (b).** An objective $D$ is *$C$-competitive* if the following holds for every $V$, $T$, teacher $p$ and class $\mathcal Q$: every $\hat q\in\arg\min_{q\in\mathcal Q}D(p,q)$, with any tie-breaking, satisfies $\mathrm{TV}(p,\hat q)\le C\cdot\mathrm{OPT}$.

**Formalising (c).** (c) cannot mean an absolute bound on the variance. With shared parameters, even log-loss BC fails that: at realizability, its teacher-forced gradient has variance $\mathrm{tr}\,I_T(\theta^\star)\asymp T$. So (c) must be relative to some normalisation. Since the paper's choice is not available, I prove (c) in three normalisation-free forms:

- **(c1) Pointwise domination.** The estimator is pointwise dominated by the log-loss BC estimator on the same sample. Any $T$-uniform bound for log-loss BC therefore transfers, under any per-token normalisation.
- **(c2) Fisher units.** $\mathrm{Var}\le K\cdot\mathrm{tr}\,I_T(\theta)$ with $K$ independent of $T$. This is the scale in which sequence-level REINFORCE with log-ratio rewards blows up by a factor $\Theta(T)$.
- **(c3) Per-token signal.** The scalar that multiplies $\nabla_\theta\log q_\theta(x_t\mid x_{<t})$ has variance bounded independently of $T$.

---

## 2. What (b) forces: the kink theorem

**Theorem 1.** Let $f$ be convex with $f(1)=0$ and $D_f\not\equiv0$.

1. $\kappa_f\,\mathrm{TV}\le D_f\le\Delta_f\,\mathrm{TV}$.
2. If $\kappa_f>0$ and $\Delta_f<\infty$, then minimising $D_f$ is $C^\star_f$-competitive with $C^\star_f=\Delta_f/\kappa_f$, for every $T$.
3. **$C^\star_f$ is sharp.** Take any $T,V$ with $\lvert V\rvert^T\ge4$ and any $\gamma>0$. There are a teacher $p$ and a class $\mathcal Q=\{q,q'\}$ with $\mathrm{TV}(p,\hat q)\ge(C^\star_f-\gamma)\,\mathrm{OPT}$.
4. **No kink, no constant.** If $\kappa_f=0$ ($f$ differentiable at $1$) or $\Delta_f=\infty$, no finite $C$ works.
   - If in addition $f''(1)>0$ exists and $\Delta_f<\infty$, there are instances with
     $$\mathrm{TV}(p,\hat q)\ge(1-o(1))\sqrt{\tfrac{\Delta_f}{2f''(1)}\,\mathrm{OPT}}\quad\text{as }\mathrm{OPT}\to0.$$
   - For JSD we have $\Delta=\ln2$ and $f''(1)=\tfrac14$, so the bound is $\sqrt{2\ln 2\cdot\mathrm{OPT}}$. It matches the upper bound $\mathrm{TV}(p,\hat q)\le\sqrt{2\,\mathrm{JSD}(p,\hat q)}\le\sqrt{2\,\mathrm{JSD}(p,q)}\le\sqrt{2\ln2\cdot\mathrm{TV}(p,q)}$, which holds for every $q\in\mathcal Q$. That upper bound uses $\mathrm{JSD}\ge\tfrac12\mathrm{TV}^2$ (Pinsker applied to both halves) and $\mathrm{JSD}\le\ln 2\cdot\mathrm{TV}$ (part 1).
5. $\Delta_f\ge\kappa_f$, with equality iff $D_f=\kappa_f\,\mathrm{TV}$. So TV is the only $f$-divergence with $C^\star=1$.

*Proof.* Write $a=f'(1^-)$ and $b=f'(1^+)$.

**(1), lower bound.** By convexity, $f(t)\ge b(t-1)$ and $f(t)\ge a(t-1)$ for every $t\ge0$. Use the first inequality on $\{p\ge q\}$ and the second on $\{p<q\}$. For atoms with $q=0$, use $f'(\infty)\ge b$. Then
$$D_f\ge b\sum(p-q)_+-a\sum(q-p)_+=(b-a)\,\mathrm{TV}.$$

**(1), upper bound.**

- The map $f\mapsto f_c:=f+c\,(t-1)$ leaves $D_f$ unchanged, because $\sum_x(p-q)=0$.
- Convexity gives $f(t)\le(1-t)f(0)$ on $[0,1]$. It also gives $f(t)\le(t-1)f'(\infty)$ on $[1,\infty)$, since chord slopes from the point $1$ increase to $f'(\infty)$.
- Hence $f_c(t)\le L(c)\,\lvert t-1\rvert$ with $L(c)=\max\{f(0)-c,\;f'(\infty)+c\}$. The same bound covers atoms with $q=0$.
- Therefore $D_f\le 2L(c)\,\mathrm{TV}$. Choosing $c=(f(0)-f'(\infty))/2$ gives $2L=\Delta_f$.

**(2).** For every $q\in\mathcal Q$,
$$\kappa\,\mathrm{TV}(p,\hat q)\le D_f(p\|\hat q)\le D_f(p\|q)\le\Delta\,\mathrm{TV}(p,q).$$

**(3).** The construction uses four sequences $a,b_1,b_2,c$. Let $m=(1-\delta)/2$ and set:

- $p=(\delta,m,m,0)$;
- $q'=(0,m,m,\delta)$, which moves mass to a fresh sequence. Here $\mathrm{TV}=\delta$ and $D_f=\Delta_f\,\delta$ exactly;
- $q_\eta=(\delta,m+\eta,m-\eta,0)$. Here $\mathrm{TV}=\eta$, and the one-sided expansions of $f$ at $1$ give $D_f=\kappa_f\,\eta+o(\eta)$.

Take $\eta=(C^\star-\gamma)\,\delta$ and let $\delta\to0$. Then $D_f(p\|q_\eta)<D_f(p\|q')$, so $\hat q=q_\eta$ and $\mathrm{TV}(p,\hat q)/\mathrm{OPT}=C^\star-\gamma$.

**(4).** Use the same construction.

- If $\kappa=0$, then $D_f(q_\eta)=o(\eta)$. So for small $\delta$, $q_\eta$ is selected whatever the ratio $\eta/\delta=M$.
- If $\Delta=\infty$, then $D_f(q')=\infty$.
- If $f''(1)$ exists, the linear terms cancel and
  $$D_f(q_\eta)=\tfrac{f''(1)}2\sum\frac{(p-q)^2}{q}+o(\eta^2)=2f''(1)\,\eta^2(1+o(1)).$$
  So $q_\eta$ wins whenever $2f''(1)\,\eta^2<\Delta\,\delta$.

**(5).** Convexity gives $f(0)\ge-a$ and $f'(\infty)\ge b$. Equality in both means $f$ is affine on $[0,1]$ and on $[1,\infty)$, that is,
$$f=\tfrac{b-a}2\lvert t-1\rvert+\tfrac{a+b}2(t-1).\qquad\square$$

**Beyond $f$-divergences.** Parts (3) and (4) use only two facts about the objective:

- it is $o(\mathrm{TV})$ along a smooth perturbation of $p$;
- it is $\ge c\,\mathrm{TV}$ along a perturbation that changes the support.

Any objective with both properties fails (b). This covers JSD-type objectives that are smooth at $q=p$, including PA-JSD under the assumption stated at the top.

**Corollary (adding a kink).** Suppose $0\le D_0\le c_0\,\mathrm{TV}$. Then $D_0+\lambda\,\mathrm{TV}$ satisfies $\lambda\,\mathrm{TV}\le D_0+\lambda\,\mathrm{TV}\le(c_0+\lambda)\,\mathrm{TV}$, so it is $(1+c_0/\lambda)$-competitive. For example, $\mathrm{JSD}+\lambda\,\mathrm{TV}$ has $C^\star=1+\ln2/\lambda$ exactly.

**Remark (token-level surrogates are not enough).** The sequence-level TV is $\le\sum_t\mathbb E\,\mathrm{TV}(p_t,q_t)$, but minimising that step-wise sum loses a factor of order $\sqrt T$. Here is an instance.

- Teacher: $p=\mathrm{Ber}(\tfrac12)^{\otimes T}$.
- Student $q_1=\mathrm{Ber}(\tfrac12+\epsilon)^{\otimes T}$: step-wise sum $T\epsilon$, but $\mathrm{TV}\approx\sqrt{2/\pi}\,\sqrt T\epsilon$.
- Student $q_2$: perturbs only the first token, by $\eta=T\epsilon/2$. Step-wise sum $\eta$, and $\mathrm{TV}=\eta$.

The surrogate prefers $q_2$, whose TV is larger by a factor of about $\sqrt T/1.6$. This is why (a), an exact sequence-level objective, matters for (b).

---

## 3. The objective and its guarantees

**Definition.** For $\beta\in(0,1]$ and $u(x)=q(x)/p(x)$,
$$\mathcal D_\beta(p\|q)=\mathbb E_p[(1-u^\beta)_+].$$
This is $D_{f_\beta}$ with $f_\beta(t)=(t-t^{1-\beta})\mathbf 1\{t\ge1\}$. Its constants are $f_\beta(0)=0$, $f_\beta'(\infty)=1$ and $\kappa=\beta$, so $C^\star=1/\beta$.

**Theorem 2.**

1. $\beta\,\mathrm{TV}\le\mathcal D_\beta\le\mathrm{TV}$. Hence every population minimiser satisfies $\mathrm{TV}(p,\hat q)\le\mathrm{OPT}/\beta$.
2. **Finite sample, offline.** Let $x_1,\dots,x_n\sim p$ i.i.d. and
   $$\hat L_n(q)=\frac1n\sum_i\big(1-\min\{1,u(x_i)^\beta\}\big).$$
   Let $\hat q_n$ be an $\varepsilon_{\rm opt}$-approximate minimiser of $\hat L_n$ over a finite $\mathcal Q$. With probability $\ge1-\delta$,
   $$\mathrm{TV}(p,\hat q_n)\le\frac1\beta\Big(\mathrm{OPT}+2\sqrt{\tfrac{\log(2\lvert\mathcal Q\rvert/\delta)}{2n}}+\varepsilon_{\rm opt}\Big).$$
   This holds with no realizability, no determinism, and no dependence on $T$ or $\lvert V\rvert$.
3. **Parametric classes.** Take a $d$-parameter class on a ball of radius $R$, with $\|\nabla_\theta\log q_\theta(v\mid h)\|\le G$. The bound in part 2 still holds after two changes: replace $\log\lvert\mathcal Q\rvert$ by $d\log(1+2\beta TGRn)$, and add $4/(\beta n)$. Only $\log T$ enters.

*Proof.*

**(1).** For $u\in[0,1]$:
- $u^\beta\ge u$, so $1-u^\beta\le1-u$;
- concavity gives $u^\beta\le1+\beta(u-1)$, so $1-u^\beta\ge\beta(1-u)$.

Take $\mathbb E_p$ and use $\mathrm{TV}=\mathbb E_p[(1-u)_+]$.

**(2).** Each loss lies in $[0,1]$. Hoeffding's inequality and a union bound give $\sup_q\lvert\hat L_n(q)-\mathcal D_\beta(q)\rvert\le\epsilon_n$. Then, for every $q\in\mathcal Q$,
$$\beta\,\mathrm{TV}(\hat q_n)\le\mathcal D_\beta(\hat q_n)\le\hat L_n(\hat q_n)+\epsilon_n\le\hat L_n(q)+\varepsilon_{\rm opt}+\epsilon_n\le\mathrm{TV}(q)+2\epsilon_n+\varepsilon_{\rm opt}.$$

**(3).** The map $z\mapsto(1-e^{\beta z})_+$ is $\beta$-Lipschitz, and $\lvert\log q_\theta(x)-\log q_{\theta'}(x)\rvert\le TG\,\|\theta-\theta'\|$. So a $1/(\beta TGn)$-net in $\theta$ is a $1/n$-net in sup-norm. $\square$

**Remarks.**

- **The procedure is offline.** Its data are the SFT data (teacher samples) plus one stored scalar, $\log p(x)$, per sequence.
- **Deterministic teachers.** Take prompts $c$ with deterministic answers $x^\star(c)$.
  - $\mathcal D_1=\mathbb E_c[1-q(x^\star(c)\mid c)]$, the expected exact-match failure.
  - Log-loss BC minimises $\mathbb E_c[-\log q(x^\star(c)\mid c)]$.
  - So TV minimisation is log-loss with $-\log u$ replaced by $1-u$. This turns horizon-freeness in the realizable case (Foster et al.) into $C=1$ agnostically.
  - Log-loss is not agnostic, even at $T=1$. Let $q_1$ be exact on a $1-\varepsilon$ fraction of prompts and put negligible mass on the answer elsewhere: its TV is $\varepsilon$ but its log-loss is unbounded. It loses to $q_2\equiv\tfrac12$, whose TV is $\tfrac12$.
- **Stochastic, misspecified teachers.** Here log-loss BC is **not** horizon-free: in §6 its factor grows like $C\approx0.16\sqrt T$ for a teacher that is a 1% mixture. This is consistent with the $\Omega(H)$ lower bound for next-token-prediction objectives of Rohatgi et al. (2025). $\mathcal D_\beta$ escapes that lower bound because it is not token-additive: one sequence-level weight couples all positions.

---

## 4. The gradient estimator

**Theorem 3.** Suppose $q_\theta(x)\ne p(x)$ for all $x$ in the support of $p$; elsewhere the same formula gives a one-sided (Clarke) subgradient. Then
$$
\nabla_\theta\mathcal D_\beta(p\|q_\theta)=-\beta\,\mathbb E_{x\sim p}\big[w_\beta(x)\,S_\theta(x)\big],
\qquad w_\beta(x)=\Big(\tfrac{q_\theta(x)}{p(x)}\Big)^{\beta}\mathbf 1\{q_\theta(x)<p(x)\}\in[0,1].
$$
The one-sample estimator $\hat g(x)=-\beta\,w_\beta(x)\,S_\theta(x)$, with $x\sim p$, is unbiased. It costs the same as the log-loss BC estimator $-S_\theta(x)$: one teacher sample with its stored log-probability, and one teacher-forced student pass. No sampling from the student is needed. Moreover:

- **(c1)** $\|\hat g(x)\|\le\beta\,\|S_\theta(x)\|$ pointwise, so $\mathbb E\|\hat g-\nabla\|^2\le\beta^2\,\mathbb E_p\|S_\theta\|^2$. With per-token normalisation and $\|s_t\|\le G$, this gives $\mathrm{Var}(\hat g/T)\le\beta^2G^2$ for all $T$.
- **(c2)** For $\beta\ge\tfrac12$,
  $$\mathbb E\|\hat g\|^2\le\beta^2\,\mathbb E_{q_\theta}\big[\|S_\theta\|^2\mathbf 1\{q_\theta<p\}\big]\le\beta^2\,\mathrm{tr}\,I_T(\theta).$$
  The importance weight clips itself, so no $\chi^2(q\|p)$ factor appears, for any teacher. That factor can be $e^{\Theta(T)}$.
- **(c3)** The per-token signal is the same scalar $\beta\,w_\beta(x)\in[0,\beta]$ at every position, so its variance is $\le\beta^2/4$.

*Proof.* $\mathcal D_\beta(\theta)=\sum_x p(x)\big(1-\min\{1,e^{\beta(\log q_\theta(x)-\log p(x))}\}\big)$ is a finite sum; differentiate it term by term.
- (c1) is immediate.
- (c2): for $\beta\ge\tfrac12$ we have $u^{2\beta}\le u$ on $[0,1]$. Hence
  $$\mathbb E_p[w_\beta^2\|S\|^2]\le\sum_{q<p}p\cdot\tfrac qp\,\|S\|^2\le\mathbb E_q\|S\|^2.\qquad\square$$

**In code** (PyTorch-style), the loss is one line and autograd returns $\hat g$:

```python
z = student_logprobs(x).sum(-1) - teacher_logprob_x   # sequence log-ratio, teacher-forced on teacher samples x
loss = (1 - torch.exp(beta * z).clamp(max=1.0)).mean() # unbiased estimate of D_beta; its gradient is -beta * w * S
```

**Proposition 4 (lower bound for on-policy score-function estimators).** Consider gradients of the form $\mathbb E_{q_\theta}[R\,S_\theta]$, for a sequence-level reward $R$. Estimate them by $\sum_t\hat A_t\,s_t$, where
$$\mathbb E[\hat A_t\mid x_{\le t}]=\mathbb E[R\mid x_{\le t}]-b_t(x_{<t})$$
for some baselines $b_t$. Then:
- $\sum_t\mathbb E\hat A_t^2\ge\mathrm{Var}_{q_\theta}(R)$.
- Equality holds for the Doob advantages $A_t=M_t-M_{t-1}$, where $M_t=\mathbb E[R\mid x_{\le t}]$.
- Plain REINFORCE, $\hat A_t\equiv R-b$, has energy $T\,\mathrm{Var}(R)$.

*Proof.* Conditional Jensen gives $\mathbb E\hat A_t^2\ge\mathbb E(M_t-b_t)^2$. Expanding,
$$\mathbb E(M_t-b_t)^2=\mathbb EA_t^2+\mathbb E(M_{t-1}-b_t)^2,$$
because $\mathbb E[A_t\mid x_{<t}]=0$ kills the cross term. Martingale increments are orthogonal, so $\sum_t\mathbb EA_t^2=\mathbb E(M_T-M_0)^2=\mathrm{Var}(R)$. $\square$

**Consequences of Proposition 4.**

- **The criterion.** Horizon-uniform per-token credit for a sequence-level score-function estimator exists **iff** $\mathrm{Var}_q(R)=O(1)$.
- **Log-ratio rewards fail.** Take $R=\log(q/p)$ (sequence-level reverse KL), or any reward that is a sum of per-token log-ratios. Then $\mathrm{Var}(R)=\Theta(T)$ at fixed per-token discrepancy, and no baseline removes it.
- **$\mathcal D_1$ passes.** Written on-policy, its reward is $R=\mathbf 1\{q>p\}$, with $\mathrm{Var}(R)\le\tfrac14$.
- **Cost of Doob credit.** Its Doob advantages can be estimated without bias from $k$ rollouts per position, with energy $\le\tfrac14+\tfrac{T}{2k}$. Taking $k=T$ costs $O(T^3)$ token evaluations. The offline estimator of Theorem 3 avoids this cost.

---

## 5. What (a)–(c) do not give you

**Proposition 5 (saturation).** If $\mathrm{TV}\le D\le C\,\mathrm{TV}$, then on $\{\theta:\mathrm{TV}(p,q_\theta)\ge1-\eta\}$ the objective takes values in $[1-\eta,\,C]$.

- For $C=1$ (TV itself) the landscape is $\eta$-flat far from the teacher.
- For $\mathcal D_\beta$ the range there is $[\beta(1-\eta),1]$, and the weights $w_\beta$ are exponentially small in the sequence log-ratio. The gradient is small, not noisy.
- So a larger constant $C$ also buys dynamic range. Practical remedies:
  - warm-start with SFT;
  - anneal $\beta$ upward;
  - use a prefix curriculum $\sum_t\omega_t\,\mathcal D_\beta(p_{\le t}\|q_{\le t})$ with $\sum_t\omega_t=1$ and $\omega_T\ge\tfrac12$. This keeps $C\le2/\beta$: the objective is $\le\mathrm{TV}(p,q)$ by data processing, and $\ge\tfrac\beta2\,\mathrm{TV}(p,q)$ from the $t=T$ term.

**Computation.** Guarantee (b) concerns global minimisers of an objective that Theorem 1 forces to be non-smooth. Rohatgi et al. (COLT 2025) prove three things for Hellinger-type approximation factors:

- $C=O(1)$ is information-theoretically achievable;
- next-token-prediction objectives suffer $C=\Omega(H)$;
- for autoregressive linear models, no efficient algorithm reaches a sub-polynomial $C$.

Their barrier is stated for Hellinger, not TV, so it does not formally apply here. It does indicate where the difficulty goes: a low-variance unbiased gradient for a horizon-free objective does not make an efficient learner. The hardness moves from the estimator into the optimisation landscape.

**Access.** $\mathcal D_\beta$ needs the teacher's sequence likelihoods, i.e. white-box distillation. With samples only (imitation learning), minimum-distance estimators are the alternative (Scheffé/Yatracos, $C=3$). They are horizon-free with $O(\log\lvert\mathcal Q\rvert/\varepsilon^2)$ samples, but they are tournaments rather than differentiable objectives.

---

## 6. Numerical checks

The scripts are in [`checks/`](checks/) and need `numpy` and `scipy`. Parts 2 and 3 use an exactly solvable model, so every expectation below is an exact sum over the count of ones; nothing is Monte Carlo except the samples in part 3. The model:

- **Teacher:** $p=0.99\,\mathrm{Ber}(0.5)^{\otimes T}+0.01\,\mathrm{Ber}(0.7)^{\otimes T}$. It is stochastic and has long-range correlations.
- **Student class:** $\{\mathrm{Ber}(\mu)^{\otimes T}\}$, which is misspecified.

**Part 1: Theorem 1** ([`check_part1.py`](checks/check_part1.py)).

- The sandwich $\kappa\,\mathrm{TV}\le D_f\le\Delta\,\mathrm{TV}$ holds on random pairs, with both constants attained.
- The worst-case factor equals $C^\star_f$ for each objective tested:

| objective | measured worst case | predicted $C^\star_f$ |
|---|---|---|
| TV | 1.0000 | 1 |
| $\mathcal D_{1/2}$ | 1.9998 | 2 |
| $\mathcal D_{1/4}$ | 3.9988 | 4 |
| JSD + 1·TV | 1.6930 | $1+\ln 2$ |

- For JSD, the selected student has $\mathrm{TV}(\hat q)$ equal to $\sqrt{2\ln2\cdot\mathrm{OPT}}$, with no bounded ratio to $\mathrm{OPT}$:

| $\mathrm{OPT}$ | $\mathrm{TV}(\hat q)$ | $\sqrt{2\ln2\cdot\mathrm{OPT}}$ | $\mathrm{TV}(\hat q)/\mathrm{OPT}$ |
|---|---|---|---|
| $10^{-3}$ | 0.03718 | 0.03723 | 37 |
| $10^{-5}$ | 0.00372 | 0.00372 | 372 |

**Part 2: agnostic factor $\mathrm{TV}(p,\hat q)/\mathrm{OPT}$ at the population optimum** ([`check_part2.py`](checks/check_part2.py)).

| $T$ | forward KL (log-loss BC) | JSD | TV | $\mathcal D_{1/2}$ |
|---|---|---|---|---|
| 10 | 1.12 | 1.11 | 1.00 | 1.00 |
| 1 000 | 5.53 | 1.00 | 1.00 | 1.00 |
| 10 000 | 16.28 | 1.00 | 1.00 | 1.00 |
| 30 000 | 27.46 | 1.00 | 1.00 | 1.00 |

Log-loss BC grows like $\sqrt T$. JSD happens to reach 1.00 on this instance, but its worst case is the square-root law from part 1.

**Part 2: gradient variance in Fisher units, $\mathrm{Var}/\mathrm{tr}\,I_T$.** Evaluated at a fixed per-token gap, $\mu=\mu_0+0.05$. For RKL and JSD, "REINFORCE" means the on-policy score-function estimator with a variance-optimal constant baseline.

| $T$ | $\mathcal D_1$ | $\mathcal D_{1/2}$ | SFT (log-loss) | RKL REINFORCE | JSD REINFORCE |
|---|---|---|---|---|---|
| 10 | 0.166 | 0.068 | 1.02 | 0.15 | 0.010 |
| 100 | 0.078 | 0.046 | 1.17 | 1.41 | 0.095 |
| 1 000 | 0.053 | 0.029 | 2.61 | 19.8 | 0.269 |

- $\mathcal D_\beta$ stays below $\beta^2$, as Theorem 3(c2) predicts.
- RKL grows like $T$, as Proposition 4 predicts.
- SFT also grows like $T$ in Fisher units under this correlated teacher. Only after per-token normalisation is it bounded, which is exactly the bound (c1) transfers to $\mathcal D_\beta$.
- At $T=10\,000$ the student is saturated ($\mathrm{TV}=1.0000$) and the $\mathcal D_\beta$ gradient vanishes, as Proposition 5 predicts.

**Part 2: Doob identity.** For $R=\mathbf 1\{q>p\}$ the Doob energy equals $\mathrm{Var}_q(R)$ exactly (0.1299 at $T=1000$), while the naive REINFORCE energy is $T\,\mathrm{Var}(R)=129.9$.

**Part 3: finite-sample ERM** ([`check_part3.py`](checks/check_part3.py)). Median of $\mathrm{TV}(p,\hat q_n)/\mathrm{OPT}$ with $n=50\,000$ teacher samples held fixed:

| $T$ | $\mathcal D_1$ ERM | $\mathcal D_{1/2}$ ERM | log-loss ERM |
|---|---|---|---|
| 100 | 1.00 | 1.00 | 2.18 |
| 1 000 | 1.00 | 1.00 | 5.45 |
| 10 000 | 1.00 | 1.00 | 16.27 |

---

## References

- D. J. Foster, A. Block, D. Misra. *Is Behavior Cloning All You Need? Understanding Horizon in Imitation Learning.* NeurIPS 2024. arXiv:2407.15007.
- D. Rohatgi, A. Block, A. Huang, A. Krishnamurthy, D. J. Foster. *Computational-Statistical Tradeoffs at the Next-Token Prediction Barrier: Autoregressive and Imitation Learning under Misspecification.* COLT 2025. arXiv:2502.12465.
- Y. Wen, Z. Li, W. Du, L. Mou. *f-Divergence Minimization for Sequence-Level Knowledge Distillation.* ACL 2023. Considers TVD distillation through a step-wise decomposition; see the remark on token-level surrogates in §2.
