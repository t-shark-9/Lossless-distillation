# The optimal agnostic exponent: the square root is the price of curvature

> **Open problem.** Proposition 4.13 gives $\mathrm{TV}(\hat\theta)=O(\sqrt{\mathrm{TV}^\star})$. Is the square
> root necessary for any smooth, tractable surrogate? Equivalently, what is the best $\phi$ with
> $\mathrm{TV}(\arg\min D)\le\phi(\mathrm{TV}^\star)$ over divergences $D$ that admit unbiased per-token
> gradients? A lower bound $\phi(t)\gtrsim\sqrt t$ for all $f$-divergences with $f''(1)>0$ would be a
> clean impossibility result.

The paper itself was not available when this was written, so the setting (§0) is reconstructed from
the statement above. Constants depend on normalisation conventions; exponents do not. The proofs are
self-contained. Every claim that can be checked on finite examples is checked by
[`verify_agnostic_exponent.py`](verify_agnostic_exponent.py) (§7).

## Answer

1. **Yes: every smooth $f$-divergence needs the square root, and the constant is exact.** Let $f$ be
   convex with $f(1)=0$ and $0<f''(1)<\infty$, and let $K_f:=f(0)+f'(\infty)\in(0,\infty]$. Then
   $$\lim_{t\to0}\frac{\phi_f(t)}{\sqrt t}=\sqrt{\frac{K_f}{2f''(1)}} .$$
   Non-asymptotically, $(1-t)\,\psi_f^{-1}(K_ft)\le\phi_f(t)\le\psi_f^{-1}(K_ft)$, where $\psi_f$ is the
   tight lower bound of $D_f$ in terms of TV (Theorem 1). For squared Hellinger this reads
   $\sqrt{2t-3t^2}\le\phi(t)\le\sqrt{2t-t^2}$. If $K_f=\infty$ (KL in either direction, $\chi^2$),
   there is no guarantee at all: $\phi_f\equiv1$. The lower-bound witness is a single token over a
   few letters (four suffice for the asymptotics) with a class of two models. It therefore covers
   sequence-level and token-level objectives, both argument orders, and every prefix distribution.

2. **The square root is the price of curvature, not of tractability.** The exponent of $\phi_f$ is
   $1/\gamma$, where $\gamma$ is the order of contact of $f$ with its tangent at $1$ (Theorem 4):
   - $\gamma=2$ ($C^2$) gives $\sqrt t$;
   - $\gamma\in(1,2)$ ($C^1$ with $f''(1)=\infty$) gives $t^{1/\gamma}$;
   - a linear rate occurs iff $f$ has a kink at $1$.

   TV itself admits an unbiased per-token gradient: teacher forcing reweighted by sequence weights in
   $[0,1)$ computed from per-token log-probs (Proposition 5). It attains $\phi(t)=t$, and no divergence
   can do better. So **the best $\phi$ over divergences with unbiased per-token gradients is the
   identity. Over surrogates that are $C^2$ at the diagonal, the best is $\Theta(\sqrt t)$.**

3. **The $\sqrt t$ is not an artefact of zero-probability tails.** Theorem 3 derives it for every
   locally quadratic divergence using only full-support models with all likelihood ratios in
   $[1-\delta,1+\delta]$. This covers any $C^2$ $f$-divergence (including KL with bounded ratios),
   $\ell_2^2$ and $\mathrm{MMD}^2$. The cause is the $\sqrt n$ distortion between $\ell_1^n$ (TV) and a
   Hilbert norm (the surrogate's Hessian) at resolution $n\approx\delta/t$.

4. **The strict reading of "per-token" only makes things worse.** The only $f$-divergences with an
   exact per-token chain rule are $\mathrm{KL}(P\Vert Q)$ (teacher prefixes) and $\mathrm{KL}(Q\Vert P)$
   (student prefixes), by Proposition 6. Both have $\phi\equiv1$. Token-level sums of bounded
   divergences also pay the horizon inside the root: $\phi\gtrsim\sqrt{T\,t}$ (Proposition 7).

## 0. Setting and notation

- $\mathcal X$ is a finite or countable set of outcomes, typically sequences $x=(x_1,\dots,x_T)$. $P$
  is the teacher and $\mathcal Q$ the model class.
- $\mathrm{TV}(P,Q)=\tfrac12\sum_x|P(x)-Q(x)|$ and $\mathrm{TV}^\star=\inf_{Q\in\mathcal Q}\mathrm{TV}(P,Q)$.
- The **agnostic distortion** of a divergence $D$ is
  $$\phi_D(t)=\sup\Bigl\{\mathrm{TV}(P,\hat Q)\;:\;P,\ \mathcal Q\text{ with }\mathrm{TV}^\star\le t,\ \
  \hat Q\in\arg\min_{Q\in\mathcal Q}D(P\Vert Q)\Bigr\}.$$
  It is the smallest $\phi$ with $\mathrm{TV}(\arg\min D)\le\phi(\mathrm{TV}^\star)$ for all teachers and
  classes. Since $\hat Q\in\mathcal Q$, always $\phi_D(t)\ge t$.
- **$f$-divergences.** Take $f:(0,\infty)\to\mathbb R$ convex with $f(1)=0$, and set
  $D_f(P\Vert Q)=\sum_xQ(x)\,f\bigl(P(x)/Q(x)\bigr)$. The boundary values are $f(0):=\lim_{u\downarrow0}f(u)$
  and $f'(\infty):=\lim_{u\to\infty}f(u)/u$, with the convention $0\cdot f(p/0):=p\,f'(\infty)$.
  - Replacing $f$ by $f+c(u-1)$ changes neither $D_f$ nor $K_f$. When convenient we therefore assume
    $f\ge0$ by subtracting a supporting line at $1$.
  - $D_f(Q\Vert P)=D_{f^\diamond}(P\Vert Q)$ with $f^\diamond(u)=uf(1/u)$. Since $K_{f^\diamond}=K_f$ and
    $(f^\diamond)''(1)=f''(1)$, every statement below holds for both argument orders.
- **Two TV-profiles** of $D_f$:
  - upper profile: $D_f\le K_f\,\mathrm{TV}$ (Lemma 1), where $K_f=f(0)+f'(\infty)$ is the range of
    $D_f$;
  - lower profile: $\psi_f(v)=\inf\{D_f(P\Vert Q):\mathrm{TV}(P,Q)=v\}$ over all finite alphabets
    (Vajda's tight lower bound), with inverse $\psi_f^{-1}(y):=\sup\{v\in[0,1]:\psi_f(v)\le y\}$.

The whole problem is the gap between these two profiles. A surrogate cannot tell a concentrated
error of TV-size $t$, which it charges at the top of its range ($K_ft$), from a diffuse error of
TV-size $v$, which it charges at the bottom ($\psi_f(v)\approx2f''(1)v^2$).

## 1. Two lemmas

**Lemma 1 (upper profile).** $D_f(P\Vert Q)\le K_f\,\mathrm{TV}(P,Q)$. Equality holds when $P$ and $Q$
disagree only at points where one of them is zero.

*Proof.* Take $f\ge0$. Convexity and $f(1)=0$ give $f(u)\le(1-u)f(0)$ on $[0,1]$. They also give
$f(u)\le(u-1)f'(\infty)$ on $[1,\infty)$, because the slope of $f$ increases to $f'(\infty)$. Hence
$$D_f=\sum_{P<Q}Qf\bigl(\tfrac PQ\bigr)+\sum_{P>Q}Qf\bigl(\tfrac PQ\bigr)\le f(0)\sum_{P<Q}(Q-P)+f'(\infty)\sum_{P>Q}(P-Q)=K_f\,\mathrm{TV}.$$
A point with $P=0<Q$ contributes exactly $Q\,f(0)$, and a point with $Q=0<P$ exactly $P\,f'(\infty)$.
This gives the equality case. $\blacksquare$

**Lemma 2 (lower profile).**

- (a) $\psi_f$ is convex and nondecreasing, with $\psi_f(0)=0$.
- (b) If $0<f''(1)<\infty$, then $\psi_f(v)=2f''(1)\,v^2\,(1+o(1))$ as $v\to0$.
- (c) Two closed forms, both tight on binary pairs:
  - squared Hellinger $H^2=1-\sum\sqrt{PQ}$ has $\psi(v)=1-\sqrt{1-v^2}$ (Le Cam's inequality);
  - triangular discrimination $\Delta=\sum\frac{(P-Q)^2}{P+Q}$ has $\psi(v)=2v^2$ (Cauchy–Schwarz).

*Proof.* (a) Take pairs $(P_1,Q_1)$ and $(P_2,Q_2)$ on disjoint alphabets. The direct sum
$(\lambda P_1\oplus(1-\lambda)P_2,\ \lambda Q_1\oplus(1-\lambda)Q_2)$ has TV and $D_f$ equal to the
$\lambda$-mixtures of those of the parts. So $\psi_f$ is convex, and being $\ge0=\psi_f(0)$ it is
nondecreasing.

(b) Normalise $f'(1)=0$.

*Upper bound.* The pair $P_0=(\tfrac12,\tfrac12)$, $Q_0=(\tfrac12+v,\tfrac12-v)$ has TV $v$ and
$D_f=\sum_\pm(\tfrac12\pm v)\,f\bigl(\tfrac1{1\pm2v}\bigr)=2f''(1)v^2+o(v^2)$.

*Lower bound.* Fix $\eta>0$ and choose $\delta>0$ with $f(1+s)\ge(\tfrac12f''(1)-\eta)s^2$ for
$|s|\le\delta$. Put $m=\min\{f(1+\delta),f(1-\delta)\}/\delta>0$. Since $s\mapsto f(1+s)/s$ is
nondecreasing, $f(1+s)\ge m|s|$ for $|s|\ge\delta$, and $f'(\infty)\ge m$.

Given $P,Q$ with TV $v$, let $r=P/Q$ and $L=\{|r-1|\le\delta\}$. Split $v=v_L+v_N$ into the TV-mass
on $L$ and off $L$. By Cauchy–Schwarz, $\sum_L(P-Q)^2/Q\ge(2v_L)^2/Q(L)\ge4v_L^2$. Therefore
$$D_f\ \ge\ (2f''(1)-4\eta)\,v_L^2+2m\,v_N\ \ge\ (2f''(1)-4\eta)\,v^2\qquad\text{whenever } v\le\frac{m}{2f''(1)-4\eta}.$$
The last step holds because the middle expression is convex in $v_L$ and decreasing on $[0,v]$. Now
let $\eta\to0$.

(c) Le Cam's inequality is $\mathrm{TV}\le\sqrt{1-(1-H^2)^2}$. For the triangular discrimination,
$(\sum|P-Q|)^2\le\Delta\cdot\sum(P+Q)$. Both are equalities for $P=(\frac{1+v}2,\frac{1-v}2)$,
$Q=(\frac{1-v}2,\frac{1+v}2)$. $\blacksquare$

## 2. Main theorem

**Theorem 1.** Let $f$ be convex with $f(1)=0$, and let $t\in(0,1)$.

- (i) **Achievability.** For every $P$, $\mathcal Q$ and $\hat Q\in\arg\min_{Q\in\mathcal Q}D_f(P\Vert Q)$,
  $$\mathrm{TV}(P,\hat Q)\ \le\ \psi_f^{-1}\bigl(K_f\,\mathrm{TV}^\star\bigr).$$
- (ii) **Impossibility.** For every $v$ with $\psi_f(v)<K_ft/(1-t)$, there are $P$ and
  $\mathcal Q=\{Q^\star,\hat Q\}$ on a finite alphabet such that $\mathrm{TV}(P,Q^\star)=t$,
  $\arg\min_{Q\in\mathcal Q}D_f(P\Vert Q)=\{\hat Q\}$, and $\mathrm{TV}(P,\hat Q)=(1-t)v$.

Consequently, whenever $\psi_f$ is continuous and increasing (for example when $f''(1)>0$),
$$(1-t)\,\psi_f^{-1}(K_ft)\ \le\ \phi_f(t)\ \le\ \psi_f^{-1}(K_ft).$$

*Proof.* (i) Take any $Q\in\mathcal Q$. The definition of $\psi_f$, minimality of $\hat Q$, and
Lemma 1 give
$$\psi_f(\mathrm{TV}(P,\hat Q))\le D_f(P\Vert\hat Q)\le D_f(P\Vert Q)\le K_f\,\mathrm{TV}(P,Q).$$
Now take the infimum over $Q$.

(ii) Choose $(P_0,Q_0)$ on an alphabet $\mathcal X_0$ with $\mathrm{TV}(P_0,Q_0)=v$ and
$D_f(P_0\Vert Q_0)<K_ft/(1-t)$. Add two fresh symbols $a,b$ and set
$$P=(1-t)P_0+t\,\delta_a,\qquad Q^\star=(1-t)P_0+t\,\delta_b,\qquad\hat Q=(1-t)Q_0+t\,\delta_a .$$

- $Q^\star$ is exact on $\mathcal X_0$ and moves mass $t$ from $a$ to $b$. So
  $\mathrm{TV}(P,Q^\star)=t$, and by the equality case of Lemma 1, $D_f(P\Vert Q^\star)=K_ft$.
- $\hat Q$ is exact on $\{a,b\}$ and carries a diffuse error on $\mathcal X_0$. So
  $\mathrm{TV}(P,\hat Q)=(1-t)v$ and $D_f(P\Vert\hat Q)=(1-t)D_f(P_0\Vert Q_0)<K_ft$. $\blacksquare$

**Corollary 2 (the impossibility result).** If $0<f''(1)<\infty$, then
$$\lim_{t\to0}\frac{\phi_f(t)}{\sqrt t}\;=\;\sqrt{\frac{K_f}{2f''(1)}}\;\in(0,\infty].$$
So $\phi_f(t)\gtrsim\sqrt t$ for every such $f$, and the rate $\sqrt t$ is attained exactly when
$K_f<\infty$.

If $K_f=\infty$, then $\phi_f\equiv1$. Let $Q^\star$ drop an atom of mass $t$ (if $f'(\infty)=\infty$)
or invent one (if $f(0)=\infty$); then $D_f(P\Vert Q^\star)=\infty$. Meanwhile
$\hat Q=(1-\eta)\delta_c+\eta P$ has finite divergence and TV close to $1$. Here $c$ is a point of
zero teacher mass if $f(0)<\infty$, and of small positive teacher mass otherwise.

*Proof.* By Lemma 2(b), $\psi_f^{-1}(y)=\sqrt{y/2f''(1)}\,(1+o(1))$. Substitute into Theorem 1.
$\blacksquare$

| surrogate $D$ | $K_f$ | $f''(1)$ | agnostic distortion $\phi_D(t)$ |
|---|---|---|---|
| squared Hellinger $1-\sum\sqrt{PQ}$ | $1$ | $1/4$ | $\sqrt{2t-3t^2}\le\phi\le\sqrt{2t-t^2}$ |
| triangular (Vincze–Le Cam) $\sum\frac{(P-Q)^2}{P+Q}$ | $2$ | $1$ | $\sqrt{t-t^2}\le\phi\le\sqrt t$ |
| Jensen–Shannon | $\log 2$ | $1/4$ | $\sqrt{2\log2\cdot t}\,(1+o(1))\approx1.18\sqrt t$ |
| skew KL $\mathrm{KL}(P\Vert\alpha P+(1-\alpha)Q)$ | $\log\frac1\alpha$ | $(1-\alpha)^2$ | $\frac1{1-\alpha}\sqrt{\frac12\log\frac1\alpha\cdot t}\,(1+o(1))$ |
| Rényi $D_\alpha$, $0<\alpha<1$ (incl. Bhattacharyya) | $1$ | $\alpha(1-\alpha)$ | $\sqrt{t/(2\alpha(1-\alpha))}\,(1+o(1))$ |
| $\mathrm{KL}(P\Vert Q)$, $\mathrm{KL}(Q\Vert P)$, $\chi^2$, Rényi $\alpha\ge1$ | $\infty$ | — | $1$ |
| TV (not smooth) | $1$ | — | $t$ |

The Rényi rows use $f(u)=1-u^\alpha$. $D_\alpha$ is an increasing function of this $f$-divergence, and
increasing transforms do not change the minimiser. For squared Hellinger, the bound
$\mathrm{TV}\le\sqrt{2\,\mathrm{TV}^\star}$ that follows from Le Cam's inequality is tight in both exponent
and constant. Among the standard bounded surrogates, triangular discrimination has the smallest
constant: $\mathrm{TV}(\arg\min\Delta)\le\sqrt{\mathrm{TV}^\star}$, tight up to a factor $\sqrt{1-t}$.

**Remarks.**

1. **Scope of the witness.** The witness is a single token over the vocabulary
   $\mathcal X_0\cup\{a,b\}$. Four letters suffice asymptotically: take $P_0=(\frac12,\frac12)$.
   - With $T=1$, sequence-level $D_f$ (either argument order) and token-level sums (any prefix
     distribution) all reduce to $D_f$ of the first-token distributions, so the witness covers them.
   - It also covers every increasing transform of $D_f$, since the minimiser is unchanged.
   - For $\varepsilon$-approximate minimisers, the upper bound becomes
     $\psi_f^{-1}(K_f\mathrm{TV}^\star+\varepsilon)$.
2. **Convex classes do not help.** Consider the segment $Q_\lambda=(1-\lambda)Q^\star+\lambda\hat Q$.
   The two errors have disjoint supports, so $\mathrm{TV}(P,Q_\lambda)=(1-\lambda)t+\lambda(1-t)v$ and
   still $\mathrm{TV}^\star=t$. The divergence splits as
   $$D_f(P\Vert Q_\lambda)=t\,A(\lambda)+(1-t)\,D_f\bigl(P_0\Vert(1-\lambda)P_0+\lambda Q_0\bigr),\qquad A(\lambda)=\lambda f(1/\lambda)+(1-\lambda)f(0),$$
   with $A$ nonincreasing. Suppose $(1-t)D_f(P_0\Vert Q_0)<t\,A(\frac12)$. Then every $\lambda<\frac12$
   has $D_f\ge tA(\frac12)>D_f(P\Vert Q_1)$, so every minimiser has $\lambda\ge\frac12$. Its TV is
   therefore at least $\frac12(1-t)v\approx\frac12\sqrt{A(\frac12)\,t/2f''(1)}$.
3. **No universal constant, only a universal exponent.** Consider Huber-smoothed TV,
   $$f_\delta(u)=\begin{cases}\dfrac{(u-1)^2}{4\delta}, & |u-1|\le\delta,\\[4pt] \dfrac12|u-1|-\dfrac\delta4, & \text{otherwise.}\end{cases}$$
   It has $K_f/2f''(1)=\delta(1-\delta/4)$, and numerically $\phi_{f_\delta}(t)\approx\max(t,\sqrt{\delta t})$.
   A smooth surrogate can mimic TV for $t\gtrsim\delta$. Below its smoothing scale, its exponent
   reverts to $\frac12$.

## 3. The square root without tails: every locally quadratic divergence

Theorem 1 uses a model with a zero. The square root survives when every model has full support and
every likelihood ratio lies in $[1-\delta,1+\delta]$. It also holds for divergences that are not
$f$-divergences.

**Theorem 3.** Let $D$ be any function of pairs of distributions, and let $U_n$ be uniform on $[n]$.
Suppose there are $\delta\in(0,1)$ and $0<c\le C$ with the following property: for each even $n$
there is a positive definite quadratic form $q_n$ on $\{\Delta\in\mathbb R^n:\sum_x\Delta_x=0\}$ with
$$c\,q_n(\Delta)\ \le\ D(U_n\Vert U_n+\Delta)\ \le\ C\,q_n(\Delta)\qquad\text{whenever }|\Delta_x|\le\delta/n\ \text{for all }x.$$
Then for every even $n>2C/c$ and $t=\delta/n$,
$$\phi_D(t)\ \ge\ \sqrt{\frac{c\,\delta}{2C}}\;\sqrt t .$$
The witness is a class of two full-support models whose likelihood ratios to $U_n$ lie in
$[1-\delta,1+\delta]$.

*Proof.* Let $h\in\{\pm1\}^n$ be a uniformly random balanced sign vector. Then
$\mathbb E[h_x^2]=1$ and $\mathbb E[h_xh_y]=-\frac1{n-1}$ for $x\ne y$. Writing
$q_n(\Delta)=\Delta^\top H\Delta$,
$$\mathbb E_h\,q_n(h)=\sum_xH_{xx}-\frac1{n-1}\sum_{x\ne y}H_{xy}=\frac n2\;\mathbb E_{x\ne y}\,q_n(e_x-e_y).$$
Hence there are a balanced $h^\ast$ and a pair $x^\ast\ne y^\ast$ with
$q_n(h^\ast)\le\frac n2\,q_n(e_{x^\ast}-e_{y^\ast})$. Put
$$Q^\star=U_n+\tfrac\delta n\,(e_{x^\ast}-e_{y^\ast}),\qquad\hat Q=U_n+s\,\tfrac\delta n\,h^\ast,\qquad s^2<\tfrac{2c}{Cn}.$$
Then $\mathrm{TV}(U_n,Q^\star)=\delta/n=t$ and $\mathrm{TV}(U_n,\hat Q)=s\delta/2>t$. Moreover
$$D(U_n\Vert\hat Q)\le Cs^2\tfrac{\delta^2}{n^2}\,q_n(h^\ast)\le Cs^2\tfrac{\delta^2}{n^2}\cdot\tfrac n2\,q_n(e_{x^\ast}-e_{y^\ast})<c\,\tfrac{\delta^2}{n^2}\,q_n(e_{x^\ast}-e_{y^\ast})\le D(U_n\Vert Q^\star).$$
Letting $s\uparrow\sqrt{2c/(Cn)}$ gives
$\mathrm{TV}(U_n,\hat Q)\to\delta\sqrt{c/(2Cn)}=\sqrt{c\delta/(2C)}\,\sqrt t$. $\blacksquare$

The hypothesis holds, for example, for:

- **Every $f$-divergence with $f\in C^2$ near $1$ and $f''(1)>0$**, in either argument order. Here
  $q_n=n\sum\Delta_x^2$ (the $\chi^2$ form), and $c,C\to f''(1)/2$ as $\delta\to0$.
  - This includes $\mathrm{KL}(P\Vert Q)$ and $\mathrm{KL}(Q\Vert P)$. On classes with bounded
    likelihood ratios $P/Q\le B$, Lemma 1's argument gives $K\le1+\log B$, so KL does have a
    square-root guarantee there, $\mathrm{TV}(\hat Q)\le\sqrt{(1+\log B)\,\mathrm{TV}^\star/2}$. Theorem 3
    shows it cannot do better.
- **$\ell_2^2$ and $\mathrm{MMD}^2$ with a strictly positive definite kernel.** These are exactly
  quadratic, with $c=C=1$.
- **The $\beta$-divergences.**

Geometrically, the surrogate is a Hilbert norm at the teacher, while TV is an $\ell_1$ norm. At
resolution $n$, an $n$-dense error has $\sqrt{n/2}$ times the $\ell_1$-mass of a 2-sparse error of
the same quadratic size. The quadratic regime extends down to resolution $n\approx\delta/t$, which
gives $\sqrt n\cdot t\approx\sqrt{\delta t}$.

## 4. What sets the exponent, and the best $\phi$

**Theorem 4.** Assume $K_f<\infty$. Suppose that, after subtracting a supporting line at $1$,
$c_1|s|^\gamma\le f(1+s)\le c_2|s|^\gamma$ for $|s|\le s_0$, with $\gamma\ge1$. Then
$\phi_f(t)\asymp t^{1/\gamma}$ as $t\to0$. In particular:

- **$\gamma=1$: a kink at $1$**, with $\kappa:=f'(1^+)-f'(1^-)>0$. Then
  $\phi_f(t)\le\frac{K_f}{\kappa}\,t$. For TV ($f=\frac12|u-1|$, $K_f=\kappa=1$) this is $\phi(t)=t$.
- **$1<\gamma<2$: $C^1$ with $f''(1)=\infty$**, e.g. $|u-1|^{3/2}$ near $1$. Then
  $t\ll\phi_f(t)\asymp t^{1/\gamma}\ll\sqrt t$.
- **$\gamma=2$: $0<f''(1)<\infty$.** Then $\phi_f(t)\asymp\sqrt t$ (Corollary 2).
- **$\gamma>2$: $f''(1)=0$.** Then $\phi_f$ is worse than $\sqrt t$.

Moreover, $\phi_f(t)=O(t)$ if and only if $f$ is not differentiable at $1$.

*Proof.* Run the proof of Lemma 2(b) with Jensen's inequality on the local part,
$\sum_LQ|r-1|^\gamma\ge(\sum_LQ|r-1|)^\gamma=(2v_L)^\gamma$. This gives
$\psi_f(v)\ge\min(c_1,m)\,v^\gamma$. The binary pair of Lemma 2(b) gives
$\psi_f(v)\le c_2(4v)^\gamma$ for $v\le\frac14$. Now apply Theorem 1.

For the kink, subtract the line through $(1,0)$ with slope $\frac12(f'(1^+)+f'(1^-))$. Then
$f(u)\ge\frac\kappa2|u-1|$, so $D_f\ge\kappa\,\mathrm{TV}$ and $\psi_f(v)\ge\kappa v$.

Conversely, if $f$ is differentiable at $1$, the binary pair gives $\psi_f(v)=o(v)$. Hence
$\psi_f^{-1}(K_ft)/t\to\infty$. $\blacksquare$

**The best $\phi$.** Every divergence has $\phi_D(t)\ge t$, and TV attains it. The only remaining
question is whether TV counts as tractable, and §5 shows that it does. This answers the
"equivalently" form of the problem:

- **Over all divergences with unbiased per-token gradients:** $\phi(t)=t$, attained by TV.
- **Over those that are $C^2$ at the diagonal:** $\Theta(\sqrt t)$. For $f$-divergences the sharp
  constant is $\sqrt{K_f/2f''(1)}$, and the exponent $\frac12$ cannot be improved (Corollary 2,
  Theorem 3).
- **In between, differentiable but not $C^2$:** exactly the exponents $1/\gamma\in(\frac12,1)$.

## 5. Tractability

**Proposition 5 (TV has an unbiased per-token gradient).** Let $Q_\theta(x)=\prod_sq_\theta(x_s\mid x_{<s})$
with $q_\theta$ differentiable. Let $\theta$ be such that $Q_\theta(x)\ne P(x)$ for all $x$; otherwise,
read the formula as a Clarke subgradient. Then
$$\nabla_\theta\,\mathrm{TV}(P,Q_\theta)=-\,\mathbb E_{x\sim P}\Bigl[w_\theta(x)\sum_{s=1}^T\nabla_\theta\log q_\theta(x_s\mid x_{<s})\Bigr],\qquad
w_\theta(x)=\frac{Q_\theta(x)}{P(x)}\,\mathbf 1\{Q_\theta(x)<P(x)\}\in[0,1).$$

*Proof.* $\mathrm{TV}=\sum_x(P(x)-Q_\theta(x))_+$, so
$\nabla\mathrm{TV}=-\sum_{x:\,Q_\theta(x)<P(x)}\nabla Q_\theta(x)=-\sum_xP(x)\,w_\theta(x)\,\nabla\log Q_\theta(x)$.
$\blacksquare$

The weight needs only
$\log Q_\theta(x)-\log P(x)=\sum_s[\log q_\theta(x_s\mid x_{<s})-\log p(x_s\mid x_{<s})]$, i.e. the
per-token log-probabilities that teacher forcing already computes. Since $0\le w_\theta<1$, the
estimator's second moment is at most that of plain teacher forcing. Tractability therefore does not
force the square root.

What TV lacks is curvature. The bounded weights that make TV robust also vanish on sequences with
$Q_\theta\ll P$, which is exactly where a poorly initialised student starts. Theorems 1 and 4 show
this trade-off is unavoidable: bounded influence ($K_f<\infty$) together with curvature at $1$ is
exactly what produces $\sqrt t$.

**Proposition 6 (an exact per-token chain rule forces KL).** Suppose that for all two-step processes
with full support,
$$D_f(P_{12}\Vert Q_{12})=D_f(P_1\Vert Q_1)+\mathbb E_{x\sim R}\,D_f\bigl(P_{2|x}\Vert Q_{2|x}\bigr).$$

- With $R=P_1$ (teacher prefixes), $f(u)=\kappa\,u\log u+k(u-1)$, i.e. $D_f=\kappa\,\mathrm{KL}(P\Vert Q)$.
- With $R=Q_1$ (student prefixes), $f(u)=-\kappa\log u+k(u-1)$, i.e. $D_f=\kappa\,\mathrm{KL}(Q\Vert P)$.

*Proof ($R=P_1$).* Make the second-step conditionals of $P$ and $Q$ agree except after one first
token $x_0$. Write $a=P_1(x_0)/Q_1(x_0)$ (any $a>0$ is attainable), $w=Q_{2|x_0}$ and
$b_y=P_{2|x_0}(y)/w_y$. The identity reduces to $\sum_yw_y\,h_a(b_y)=0$, where
$h_a(b)=f(ab)-f(a)-af(b)$. This must hold for every probability vector $w$ and every $b>0$ with
$\sum_yw_yb_y=1$.

Taking two-point $w$ with $b_1<1<b_2$ gives $h_a(b_1)/(b_1-1)=h_a(b_2)/(b_2-1)$. Hence
$h_a(b)=c_a(b-1)$, that is,
$$f(ab)=f(a)+af(b)+c_a(b-1).$$
Exchanging $a$ and $b$ and subtracting gives $(b-1)(c_a-f(a))=(a-1)(c_b-f(b))$. So
$c_a=f(a)-k(a-1)$ for a constant $k$. Then $F(u)=f(u)-k(u-1)$ satisfies $F(ab)=bF(a)+aF(b)$, which
means $G(u)=F(u)/u$ satisfies $G(ab)=G(a)+G(b)$. Since $f$ is convex, it is continuous, so
$G=\kappa\log u$.

The case $R=Q_1$ is identical with $h_a(b)=f(ab)-f(a)-f(b)$, which leads to $F(ab)=F(a)+F(b)$ and
$F=-\kappa\log u$. $\blacksquare$

So within $f$-divergences, "exactly per-token decomposable" and "has an agnostic guarantee" are
incompatible: the decomposable ones have $K_f=\infty$ and $\phi\equiv1$.

**Proposition 7 (token-level surrogates: the horizon enters inside the root).** Let
$$L_R(Q)=\mathbb E_{x\sim R}\sum_{s=1}^TD_f\bigl(P(\cdot\mid x_{<s})\Vert Q(\cdot\mid x_{<s})\bigr),\qquad R\in\{P,\ Q,\ \lambda P+(1-\lambda)Q\},$$
and let $\phi_L$ be its distortion with respect to sequence-level TV. Then
$$\phi_L(t)\ \ge\ (1-t)\,\psi_f^{-1}(K_f\,Tt)\ \approx\ \sqrt{\frac{T\,K_f\,t}{2f''(1)}}\qquad(TK_ft\to0).$$
For teacher forcing ($R=P$), $\phi_L(t)\le T\,\psi_f^{-1}(2K_ft)$. For token-level TV with teacher
forcing, $Tt\le\phi_L(t)\le2Tt$ whenever $Tt\le1-t$.

*Proof.* **Lower bound.** Start from the witness of Theorem 1 on the first token. Let every
continuation be a fixed token $c$, with two exceptions:

- after $a$, $Q^\star$ continues with $d\ne c$ (a prefix $Q^\star$ never produces, but $P$ produces
  with probability $t$);
- after $b$, the teacher continues with $d$ (a prefix only $Q^\star$ produces).

Neither exception changes the sequence-level TVs, which remain $t$ and $(1-t)v$. But every later step
on those prefixes costs $D_f(\delta_c\Vert\delta_d)=K_f$, so $L_R(Q^\star)=K_ft+(T-1)K_ft$, while
$L_R(\hat Q)=(1-t)D_f(P_0\Vert Q_0)$.

**Upper bound.** The hybrid argument gives $\mathrm{TV}(P,Q)\le\sum_s\mathbb E_P\mathrm{TV}(P_s,Q_s)$ and
$\mathbb E_P\mathrm{TV}(P_s,Q_s)\le2\,\mathrm{TV}(P,Q)$. Combine these with Lemma 1, the concavity of
$\psi_f^{-1}$, and the minimality of $\hat Q$. $\blacksquare$

The $\sqrt T$ gap between the teacher-forcing bounds was not closed here.

## 6. Consequences for Proposition 4.13

- The exponent $\frac12$ is optimal for every smooth surrogate. For squared Hellinger, the constant
  is optimal as well.
- Improvements can come only from three places:
  - **Constants**, through $K_f/2f''(1)$. Triangular discrimination gives
    $\mathrm{TV}\le\sqrt{\mathrm{TV}^\star}$.
  - **Non-smooth surrogates**: TV itself, or Huber-TV at a scale matched to the target accuracy.
  - **Assumptions on $\mathcal Q$** beyond convexity and bounded likelihood ratios, since
    Remark 2 and Theorem 3 show those two are not enough.
- In the realisable case ($\mathrm{TV}^\star=0$), every divergence above gives $\mathrm{TV}(\hat Q)=0$.
  The results concern only how fast losslessness degrades.

## 7. Numerical verification

`python3 notes/verify_agnostic_exponent.py` runs in about 10 s with numpy and scipy, and checks:

1. $\psi_f(v)/(2f''(1)v^2)\to1$ for Hellinger, triangular, JS, skew-KL, and $\alpha$-divergences,
   plus the two closed forms.
2. Theorem 1:
   - the explicit witness really makes the far model the $D_f$-minimiser;
   - $\phi_f(t)/\sqrt t\to\sqrt{K_f/2f''(1)}$ from both sides;
   - a randomised search over 3,000 teachers and classes finds no violation of the upper bound;
   - KL, reverse KL and $\chi^2$ have $\phi\approx1$.
3. Theorem 3 with $\delta=\frac12$: $\mathrm{TV}(\arg\min)/\mathrm{TV}^\star\approx\sqrt{n/2}$ for KL (both
   directions), $\chi^2$, Hellinger, JS, triangular and $\ell_2^2$.
4. Theorem 4: fitted exponents $0.800,\,0.667,\,0.500,\,0.333$ for
   $\gamma=1.25,\,1.5,\,2,\,3$, and $\phi_{\mathrm{TV}}(t)=t$.
5. Proposition 6: the chain rule is exact (error $10^{-15}$) for KL with teacher prefixes and reverse
   KL with student prefixes, and fails for JS, Hellinger and triangular.
6. Proposition 5: the expected estimator matches finite differences to $10^{-10}$ on a 3-token
   softmax student, with weights in $[0,1)$.
7. Proposition 7: $\mathrm{TV}(\hat Q)=\sqrt{TK_ft/2f''(1)}$ for $T=1,4,16,64$.
8. Remark 2: over the mixture segment, the minimiser has $\lambda\approx0.87$–$0.90$ and
   $\mathrm{TV}\approx0.5$–$0.7\sqrt t$.

## References (context only; the proofs above are self-contained)

- I. Vajda. *On the f-divergence and singularity of probability measures.* Periodica Math. Hungarica,
  1972. (Tight lower bounds of $D_f$ in terms of variation.)
- P. Harremoës, I. Vajda. *On pairs of f-divergences and their joint range.* IEEE Trans. Inf. Theory,
  2011. (The joint range is the convex hull of its binary part; used only to compute $\psi_f$
  numerically.)
- L. Le Cam. *Asymptotic Methods in Statistical Decision Theory.* Springer, 1986. (Hellinger vs. TV.)
