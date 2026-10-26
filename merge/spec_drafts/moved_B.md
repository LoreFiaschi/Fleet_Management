# Blocks moved out of fragments_B.tex (writer B)

Status: done (26 Oct 2026). 52 blocks. Source: fragments_B.tex as it stood before the split into spec and
scoping (copy: scratchpad `spec_B/fragments_B_full_before_split.tex`, 1149 lines). Each block is verbatim LaTeX.
Where only part of a block left the spec, "Spec keeps" names what stays (rewritten in the spec style).
Checks behind the numbers: `checks/B_gamma.py`, `checks/B_ig.py`, `checks/B_rainflow.py` and their outputs.

### M-B-1: Gaussian model paragraph, with the remark on negative damage

- From: Gaussian Model (sec:dyn_gaussian), paragraph Model, lines 10-20 of the frozen copy.
- Kind: remark.
- Suggested destination: Literature review: damage-accumulation processes.
- Spec keeps: the model laws and Assumption N1, one line each; the remark on negative damage goes.

```latex
\textbf{Model.} A mission increment is Gaussian,
$\mathcal{N}(\mu^{\inc}_{jk}, v^{\inc}_{jk})$ (inputs \texttt{mu} and \texttt{v});
the initial damage is $\mathcal{N}(\mu_0, v_0)$ and each replacement draw is
$\mathcal{N}(\mu^{\new}, v^{\new})$. A known damage has variance 0.
\textbf{Assumption N1 (independence).} The increments, the initial damage and the
replacement draws are mutually independent. A repair acts on the damage path by
the law of the cell (Section~\ref{sec:repair_laws}). The Gaussian law gives
negative damage a small probability; the input checks of
Section~\ref{sec:consistency_checks} ($\mu \geq 3\sqrt{v}$ for the initial
damage and for each increment) keep it at most $\Phi(-3) = 0.135\,\%$ for each of
these terms.
```

### M-B-2: Gaussian: the state stays Gaussian under both laws (ARD1 exact)

- From: Gaussian Model (sec:dyn_gaussian), prop:g_exact, lines 45-65 of the frozen copy.
- Kind: statement and proof.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\begin{proposition}[The state stays Gaussian]
\label{prop:g_exact}
Under N1, on every day of a schedule $D_k \sim \mathcal{N}(m_k, V_k)$, where
$(m_k, V_k)$ follow the maps~\eqref{eq:g_map_mu} and~\eqref{eq:g_map_v}, with
anchors equal to the mean and the variance just after the last intervention.
\end{proposition}

\begin{proof}
A mission adds an independent Gaussian term, and $\ARDinf$ scales the damage by
$1-\rho$; both keep the law Gaussian and give the first two cases. Under
$\ARDone$, $D^- = G + \Delta$, with $G$ the damage just after the last
intervention and $\Delta$ the increments added since; $G$ and $\Delta$ are
independent Gaussian variables by N1. Then $D^+ = G + (1-\rho)\Delta$ is Gaussian,
with mean $m_G + (1-\rho)(m^- - m_G)$ and variance
\[
  V_G + (1-\rho)^2\, V_\Delta = V_G + (1-\rho)^2\,(V^- - V_G)
  = (1-\rho)^2\, V^- + \bigl(1 - (1-\rho)^2\bigr)\, V_G .
\]
A replacement starts from the replacement draw. Therefore the law is Gaussian on
every day, with the moments of the maps.
\end{proof}
```

### M-B-3: Gaussian: ARD1 is exact; definition of V_max

- From: Gaussian Model (sec:dyn_gaussian), lines 67-70 of the frozen copy.
- Kind: remark.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: the definitions V_max = tau^2/kappa_G^2, kappa_G = Phi^{-1}(1-eps).

```latex
So $\ARDone$ is exact for this model. Its variance map uses the factor
$(1-\rho)^2$ and the variance anchor $g^{v}$. The rows are the generic
rows of Section~\ref{sec:encoding}, with the data below;
$V_{\max} = \tau^2/\kappa_G^2$, with $\kappa_G = \Phi^{-1}(1-\varepsilon)$.
```

### M-B-4: Gaussian: derivation of the reliability row

- From: Gaussian Model, Reliability row (eq:g_row), lines 85-93 of the frozen copy.
- Kind: derivation.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: the row eq:g_row and the requirement eps < 0.5.

```latex
Gaussian cells need $\varepsilon < 0.5$, so that $\kappa_G = \Phi^{-1}(1-\varepsilon) > 0$.
For $D \sim \mathcal{N}(m, V)$,
$\Pr(D > \tau) = 1 - \Phi\bigl((\tau - m)/\sqrt{V}\bigr) \leq \varepsilon$ exactly
when $m + \kappa_G\sqrt{V} \leq \tau$. With $\kappa_G > 0$ and $\tau - \mu \geq 0$ this is
equivalent to the squared row, imposed on every day $k = 1, \ldots, T$:
\begin{equation}
  \kappa_G^2\, v_k \leq (\tau - \mu_k)^2, \qquad \mu_k \leq \tau.
  \label{eq:g_row}
\end{equation}
```

### M-B-5: Gaussian: validity of the reliability row

- From: Gaussian Model, Reliability row, lines 95-101 of the frozen copy.
- Kind: proof.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\emph{Validity.} Let $(m_k, V_k) \leq (\mu_k, v_k)$ be the true moments
(Proposition~\ref{prop:g_exact} and Section~\ref{sec:encoding}). If $V_k = 0$,
then $D_k = m_k \leq \tau$. Otherwise $\mu_k < \tau$, because $\mu_k = \tau$
forces $v_k = 0$, and
$\Pr(D_k > \tau) = 1 - \Phi\bigl((\tau - m_k)/\sqrt{V_k}\bigr)
\leq 1 - \Phi\bigl((\tau - \mu_k)/\sqrt{v_k}\bigr) \leq \varepsilon$, because
$(\tau - \mu)/\sqrt{v}$ decreases in $\mu$ and in $v$.
```

### M-B-6: Gaussian: forms, with the nonconvexity example

- From: Gaussian Model, Reliability row, Forms, lines 103-113 of the frozen copy.
- Kind: example.
- Suggested destination: Implementation details.
- Spec keeps: the exact form is a nonconvex quadratic constraint (NonConvex = 2); f, chi = v, side row mu <= tau.

```latex
\textbf{Forms.} In the form \texttt{exact}, Eq.~\eqref{eq:g_row} is a
\emph{nonconvex} quadratic constraint: $(\mu, v) = (\tau, 0)$ and
$(0, V_{\max})$ satisfy it, and their midpoint $(\tau/2, V_{\max}/2)$ does not,
since $\tau^2/2 > \tau^2/4$. The solver handles it through Gurobi's nonconvex
quadratic support (parameter \texttt{NonConvex} $= 2$). The row has the form
$v_k \leq f(\mu_k)$ with the convex function
\[
  f(\mu) = \frac{(\tau - \mu)^2}{\kappa_G^2}, \qquad \text{side row } \mu \leq \tau,
\]
so the forms \texttt{tangent} and \texttt{pwl} of
Section~\ref{sec:linear_forms} apply, and both keep the side row.
```

### M-B-7: Gaussian: remark on the level eps and the range V_max

- From: Gaussian Model, Reliability row, lines 115-127 of the frozen copy.
- Kind: rationale.
- Suggested destination: Implementation details.
- Spec keeps: V_max as the natural range of v (rows table); eps < 0.5 as a requirement.

```latex
\emph{Remark (the level $\varepsilon$ and the range $V_{\max}$).}
$V_{\max} = f(0) = \tau^2/\kappa_G^2$ is the largest variance that the row admits, at
$\mu = 0$, so it is the natural range of $v$ and the big-M constant of its rows;
the input checks of Section~\ref{sec:consistency_checks} keep $v_0$, its starting
anchor and $v^{\new}$ inside it.
$V_{\max}/\tau^2$ is $0.18$ at $\varepsilon = 0.01$, $0.61$ at $0.1$, $3.6$ at
$0.3$ and about $1600$ at $0.49$. It grows without bound as $\varepsilon \to 0.5$,
and at $\varepsilon \geq 0.5$ the squared row does not describe the reliability
requirement, because $\kappa_G \leq 0$. Hence the condition $\varepsilon < 0.5$
(Section~\ref{sec:consistency_checks}); the range is large only close to 0.5.
The variance ranges of the rainflow rows behave alike at the other end: $V^{\mathrm{C}}$,
$V^{\mathrm{B}}$ and $2\tau^2/\kappa$ grow without bound as $\varepsilon \to 1$
(Section~\ref{sec:dyn_rainflow}).
```

### M-B-8: Gaussian: facts (M1)-(M3)

- From: Gaussian Model, lines 132-143 of the frozen copy.
- Kind: statement.
- Suggested destination: Stochastic orderings: loop constraint for each degradation model.
- Spec keeps: nothing.

```latex
\textbf{Facts (M1)--(M3) of Section~\ref{sec:encoding}.}
(M1) The repair factors $1-\rho$ and $(1-\rho)^2$ lie in $[0, 1]$, and
$\delta\mu_k, \delta v_k \geq 0$.
(M2) $\theta_0 \leq \xi_0$: the initial values are the moments of the initial
damage, and each starting anchor is at least the moment of the part of it present
just after the last intervention before day 1 (Section~\ref{sec:repair_laws}). On
every later day the true moments equal the maps applied to the true statistics of
the day before (Proposition~\ref{prop:g_exact}).
(M3) If values $(\mu, v)$ that satisfy Eq.~\eqref{eq:g_row} are at least the true
moments of day $k$, then $\Pr(D_k > \tau) \leq \varepsilon$ (the validity
paragraph). The loop row of Section~\ref{sec:loop} covers $\mu$, $v$ and, under
$\ARDone$, $g^{\mu}$ and $g^{v}$.
```

### M-B-9: Inverse Gaussian: model paragraph with the IG density

- From: Inverse Gaussian Model (sec:dyn_ig), paragraph Model, lines 154-169 of the frozen copy.
- Kind: remark.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: each term of mean a has the law a Z, Z ~ IG(1, 1/eta); the density of Z for quadrature; no independence needed.

```latex
\textbf{Model.} $\mathrm{IG}(\mu, \lambda)$ is the inverse Gaussian law with mean
$\mu > 0$ and shape $\lambda > 0$. Its density is
\[
  f(x; \mu, \lambda) = \sqrt{\frac{\lambda}{2\pi x^3}}\,
  \exp\!\Bigl(-\frac{\lambda\,(x - \mu)^2}{2\mu^2 x}\Bigr), \qquad x > 0,
\]
and its variance is $\mu^3/\lambda$. For $c > 0$,
$c\,\mathrm{IG}(\mu, \lambda) = \mathrm{IG}(c\mu, c\lambda)$. The cell has one
dispersion $\eta > 0$ (input \texttt{eta}), and a term of mean $a$ has the law
$\mathrm{IG}(a, a/\eta) = a\,Z$ with $Z \sim \mathrm{IG}(1, 1/\eta)$. Its density is
$\sqrt{a/(2\pi\eta x^3)}\,\exp\bigl(-(x - a)^2/(2\eta a x)\bigr)$, and $\eta$ is
its squared coefficient of variation. Every term has this law with its own mean:
a mission increment (mean $\mu^{\inc}_{jk}$), the initial damage ($\mu_0$) and
each replacement draw ($\mu^{\new}$). A known damage $a$ is the term $a \cdot 1$.
The terms may depend on each other in any way: this model needs no independence
assumption.
```

### M-B-10: Inverse Gaussian: the damage as a weighted sum of terms, under both laws

- From: Inverse Gaussian Model, Reliability row, lines 198-208 of the frozen copy.
- Kind: derivation.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
Between replacements, the damage on day $k$ is
$D_k = \sum_t w_t X_t = \sum_t (w_t a_t)\, Z_t$, where $X_t = a_t Z_t$ are the
terms added since the last replacement (or day 0), and each $Z_t$ has the law of
$Z$ or equals 1. Each weight $w_t \in [0, 1]$ is a product of factors $1-\rho$,
and a term of weight 0 (possible when $\rho = 1$) is dropped:
$\ARDinf$ multiplies every weight by $1-\rho$, and $\ARDone$, with
$D^+ = G + (1-\rho)\,\Delta$, multiplies only the weights of the terms added since
the last intervention. When the starting anchor $g^{\mu}_0$ is below $\mu_0$, the
initial damage $\mu_0 Z_0$ is read as the two terms $g^{\mu}_0 Z_0$ and
$(\mu_0 - g^{\mu}_0) Z_0$. The mean $\sum_t w_t a_t$ is at most the tracked mean
$\mu_k$ (Section~\ref{sec:encoding}).
```

### M-B-11: Inverse Gaussian: stop-loss order (any repair weights, any dependence)

- From: Inverse Gaussian Model, prop:ig_order, lines 210-230 of the frozen copy.
- Kind: statement and proof.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\begin{proposition}[Stop-loss order]
\label{prop:ig_order}
Let $D = \sum_t w_t a_t Z_t$ with repair weights $w_t \in [0, 1]$, means
$a_t \geq 0$, $\sum_t w_t a_t \leq \mu$, and each $Z_t$ either with the law of $Z$
or equal to 1, under any joint law. Then
$\mathbb{E}(D - d)_+ \leq \mathbb{E}(\mu Z - d)_+$ for every $d \geq 0$.
\end{proposition}

\begin{proof}
Let $c_t = w_t a_t$ and $A = \sum_t c_t$. If $A = 0$, then $D = 0$. Otherwise, $x \mapsto (x - d)_+$
is convex and nondecreasing, $Z \geq 0$, and
$(A - d)_+ \leq \mathbb{E}(AZ - d)_+$ by Jensen's inequality ($\mathbb{E}Z = 1$),
so
\[
  \mathbb{E}(D - d)_+
  = \mathbb{E}\Bigl(\sum_t \frac{c_t}{A}\,(A Z_t) - d\Bigr)_+
  \leq \sum_t \frac{c_t}{A}\,\mathbb{E}(A Z_t - d)_+
  \leq \mathbb{E}(A Z - d)_+
  \leq \mathbb{E}(\mu Z - d)_+ .
\]
\end{proof}
```

### M-B-12: Inverse Gaussian: the CVaR cap theorem and its application to the row

- From: Inverse Gaussian Model, thm:ig_cap, lines 241-264 of the frozen copy.
- Kind: statement and proof.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: the row is linear, built exact, reliability_impl ignored.

```latex
\begin{theorem}[CVaR cap]
\label{thm:ig_cap}
Let $D \geq 0$ satisfy $\mathbb{E}(D - d)_+ \leq \mathbb{E}(\mu Z - d)_+$ for every
$d \geq 0$, with $0 \leq \mu \leq \bar\mu$. Then $\Pr(D > \tau) \leq \varepsilon$.
\end{theorem}

\begin{proof}
If $\mu = 0$, then $\mathbb{E}D \leq 0$ and $D = 0$. Let $\mu > 0$ and $d = \mu q$.
Since $C > q$, $d < \mu C \leq \tau$. For every $x$,
$\mathbf{1}\{x > \tau\} \leq (x - d)_+/(\tau - d)$, and
$\mathbb{E}(Z - q)_+ = \varepsilon\,(C - q)$. Then
\[
  \Pr(D > \tau) \leq \frac{\mathbb{E}(D - d)_+}{\tau - d}
  \leq \frac{\mu\,\mathbb{E}(Z - q)_+}{\tau - \mu q}
  = \frac{\varepsilon\,\mu\,(C - q)}{\tau - \mu q} \leq \varepsilon,
\]
where the last step is $\mu C \leq \tau$.
\end{proof}

By Proposition~\ref{prop:ig_order} with $\mu = \mu_k$ and
Theorem~\ref{thm:ig_cap}, the row~\eqref{eq:ig_row} gives
$\Pr(D_k > \tau) \leq \varepsilon$ on every day, under both laws. The row is
linear: it is built in its exact form, and a value of \texttt{reliability\_impl}
at the cell is ignored (Section~\ref{sec:linear_forms}).
```

### M-B-13: Inverse Gaussian: derivation of the closed form of the tail mean

- From: Inverse Gaussian Model, Computation of bar mu, item 2, lines 280-285 of the frozen copy.
- Kind: derivation.
- Suggested destination: Implementation details.
- Spec keeps: the closed form E[Z; Z > q] = F(1/q) and bar mu = tau eps / F(1/q).

```latex
  \item The tail mean has the closed form
        $\mathbb{E}[Z;\, Z > q] = F(1/q)$, so
        $\bar\mu = \tau\,\varepsilon/F(1/q)$. Derivation: substituting into the
        density gives $z\, f_Z(z) = z^{-2} f_Z(1/z)$, and the change of variable
        $u = 1/z$ gives
        $\int_q^\infty z\, f_Z(z)\, dz = \int_0^{1/q} f_Z(u)\, du = F(1/q)$.
```

### M-B-14: Inverse Gaussian: table of bar mu / tau

- From: Inverse Gaussian Model, Computation of bar mu, lines 287-298 of the frozen copy.
- Kind: example.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
Values of $\bar\mu/\tau$:
\begin{center}
\begin{tabular}{@{}lccccc@{}}
\toprule
 & $\eta = 0.05$ & $0.25$ & $1$ & $4$ & $10$ \\
\midrule
$\varepsilon = 0.001$ & 0.491 & 0.242 & 0.101 & 0.0370 & 0.0190 \\
$\varepsilon = 0.01$ & 0.570 & 0.322 & 0.156 & 0.0679 & 0.0401 \\
$\varepsilon = 0.1$ & 0.693 & 0.477 & 0.299 & 0.183 & 0.141 \\
\bottomrule
\end{tabular}
\end{center}
```

### M-B-15: Inverse Gaussian: the largest cap for the stop-loss order

- From: Inverse Gaussian Model, remark, lines 300-308 of the frozen copy.
- Kind: remark.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\emph{Remark (the largest cap for this order).} Let $\mu > \bar\mu$. For a tail
probability $\varepsilon' > \varepsilon$ close to $\varepsilon$, with
$(1-\varepsilon')$-quantile $q'$, still $\mu\,\mathbb{E}[Z \mid Z > q'] > \tau$.
The two-point law
$D = \mu\,\mathbb{E}\bigl[Z \mid \mathbf{1}\{Z > q'\}\bigr]$ is below $\mu Z$ in
the stop-loss order, by Jensen's inequality, and $\Pr(D > \tau) \geq \varepsilon'$.
So no larger cap makes Theorem~\ref{thm:ig_cap} hold for every law below $\mu Z$
in the stop-loss order, which is its hypothesis. The schedules produce only some
of these laws, so the row can reject some safe schedules.
```

### M-B-16: Inverse Gaussian: the quantile cap is not the row

- From: Inverse Gaussian Model, remark, lines 310-317 of the frozen copy.
- Kind: example.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\emph{Remark (the quantile cap is not the row).} The cap $\tau/q$ makes the model
law $\mu Z$ meet the target exactly: $\Pr(\mu Z > \tau) = \varepsilon$ at
$\mu = \tau/q$. A sum of terms below $\mu Z$ in the stop-loss order can have a
larger tail. Example: $\eta = 10$, $\varepsilon = 0.1$, so $\tau/q = 0.473\,\tau$;
three independent missions of equal mean, with total mean $\tau/q$, give
$\Pr(D > \tau) = 0.118$. The CVaR cap is $0.141\,\tau$ there. At
$\varepsilon = 0.01$ and $\eta$ from 0.05 to 10, $\bar\mu$ is 0.93 to 0.59 times
$\tau/q$.
```

### M-B-17: Inverse Gaussian: facts (M1)-(M3) and the repetition note

- From: Inverse Gaussian Model, lines 319-330 of the frozen copy.
- Kind: statement.
- Suggested destination: Stochastic orderings: loop constraint for each degradation model.
- Spec keeps: nothing.

```latex
\textbf{Facts (M1)--(M3) of Section~\ref{sec:encoding}.}
(M1) The repair factor $1-\rho$ lies in $[0, 1]$, and $\delta\mu_k \geq 0$.
(M2) $\theta_0 \leq \xi_0$, as in Section~\ref{sec:dyn_gaussian}; on every later
day the true mean equals the map~\eqref{eq:ig_map} applied to the true statistics
of the day before.
(M3) If a value $\mu \leq \bar\mu$ is at least the true mean of day $k$, then
$\Pr(D_k > \tau) \leq \varepsilon$ (Proposition~\ref{prop:ig_order} with this
$\mu$, then Theorem~\ref{thm:ig_cap}). The loop row of Section~\ref{sec:loop}
covers $\mu$ and, under $\ARDone$, $g^{\mu}$.
Each day of each repetition has its own set of terms; the argument above holds
for each of them, because it needs neither independence nor a fixed set of
terms.
```

### M-B-18: Wiener: model paragraph, with the reason for one sigma per cell

- From: Wiener Model (sec:dyn_wiener), paragraph Model, lines 341-348 of the frozen copy.
- Kind: rationale.
- Suggested destination: Literature review: damage-accumulation processes.
- Spec keeps: the model laws and Assumption W1, one line each.

```latex
\textbf{Model.} The damage is a Wiener process with drift, observed once a day. A
mission day adds a Gaussian increment $\mathcal{N}(\mu^{\inc}_{jk}, \sigma^2)$:
the mean comes from the input \texttt{mu}, and the diffusion constant
$\sigma > 0$ (input \texttt{sigma}) is the same for every mission, because it
describes the material of the component, not the route. The initial damage is
$\mathcal{N}(\mu_0, v_0)$ with $v_0 \geq 0$, and each replacement draw is
$\mathcal{N}(\mu^{\new}, v^{\new})$. \textbf{Assumption W1 (independence).} These
terms are mutually independent.
```

### M-B-19: Wiener: repairs act on the mean only (shift of the damage)

- From: Wiener Model, lines 350-354 of the frozen copy.
- Kind: remark.
- Suggested destination: Maintenance strategies: connection to the fleet management formulation.
- Spec keeps: a repair acts on the mean only; the variance falls only at a replacement.

```latex
\textbf{Repairs act on the mean only} (a modelling choice). A repair shifts the
damage by a known amount: $D^+ = D^- - \rho\, m^-$ under $\ARDinf$ and
$D^+ = D^- - \rho\,(m^- - g^{\mu})$ under $\ARDone$, with $m^- = \mathbb{E}D^-$.
The law stays Gaussian, the variance is unchanged, and only a replacement lowers
it.
```

### M-B-20: Wiener: the damage stays Gaussian

- From: Wiener Model, lines 369-371 of the frozen copy.
- Kind: statement.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
Under W1 the damage is $\mathcal{N}(m_k, V_k)$ on every day, with $(m_k, V_k)$
given by these maps: a mission adds an independent Gaussian term, and a repair or
a replacement acts as stated above.
```

### M-B-21: Wiener: a repair of the path

- From: Wiener Model, remark, lines 394-399 of the frozen copy.
- Kind: comparison.
- Suggested destination: Maintenance strategies: connection to the fleet management formulation.
- Spec keeps: nothing.

```latex
\emph{Remark (a repair of the path).} A repair of the damage path, as in the
Gaussian model, gives the same mean and the variance $(1-\rho)^2 V$ under
$\ARDinf$, or $(1-\rho)^2 V + (1 - (1-\rho)^2)\,V_G$ under $\ARDone$, with
$V_G \leq V$ the variance just after the last intervention. Both are at most $V$.
The reliability row increases in $v$, so the rows of this section are safe for
that reading as well.
```

### M-B-22: Wiener: reliability row by reference to the Gaussian section

- From: Wiener Model, Reliability row, lines 403-408 of the frozen copy.
- Kind: remark.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: the Gaussian row, its forms and the optional row, by reference.

```latex
The row is the Gaussian row~\eqref{eq:g_row}, with $\kappa_G = \Phi^{-1}(1-\varepsilon)$
and $\varepsilon < 0.5$, on every day $k = 1, \ldots, T$; its validity argument,
its forms (\texttt{exact}, \texttt{tangent}, \texttt{pwl}, with the same
function $f$ and side row) and the remark on $V_{\max} = \tau^2/\kappa_G^2$ are those of
Section~\ref{sec:dyn_gaussian}. If \texttt{v\_max\_user} is given for the cell,
the row $v_k \leq V^{\mathrm{user}}$ is added on every day.
```

### M-B-23: Wiener: facts (M1)-(M3)

- From: Wiener Model, lines 410-417 of the frozen copy.
- Kind: statement.
- Suggested destination: Stochastic orderings: loop constraint for each degradation model.
- Spec keeps: nothing.

```latex
\textbf{Facts (M1)--(M3) of Section~\ref{sec:encoding}.}
(M1) The repair factors $1-\rho$ (mean) and $1$ (variance) lie in $[0, 1]$, and
$\delta\mu_k, \delta v_k \geq 0$.
(M2) $\theta_0 \leq \xi_0$, as in Section~\ref{sec:dyn_gaussian}; on every later
day the true moments equal the maps applied to the true statistics of the day
before.
(M3) As for the Gaussian model. The loop row of Section~\ref{sec:loop} covers
$\mu$, $v$ and, under $\ARDone$, $g^{\mu}$.
```

### M-B-24: Wiener: the loop forces a replacement

- From: Wiener Model, remark, lines 419-426 of the frozen copy.
- Kind: derivation.
- Suggested destination: Stochastic orderings: loop constraint for each degradation model.
- Spec keeps: one sentence: the loop row forces a replacement in the operating phase when the train flies there.

```latex
\emph{Remark (the loop forces a replacement).} The variance grows by $\sigma^2$
on every mission day and falls only at a replacement. If the train flies $n$
missions in the operating phase and the component is not replaced there, then
$v_T \geq v_{H_1} + n\,\sigma^2 > v_{H_1}$, and the loop row $v_T \leq v_{H_1}$
fails. So every feasible schedule includes a replacement of the component in the
operating phase whenever its train flies a mission there. No input check enforces this. With
\texttt{allow\_replacement} false, the train of a Wiener cell cannot fly in the
operating phase.
```

### M-B-25: Gamma: repair on the damage path; the damage is a weighted sum

- From: Gamma Model (sec:dyn_gamma), lines 457-462 of the frozen copy.
- Kind: remark.
- Suggested destination: Maintenance strategies: connection to the fleet management formulation.
- Spec keeps: nothing.

```latex
A repair acts on the damage path by the law of the cell
(Section~\ref{sec:repair_laws}): $\ARDinf$ gives $D^+ = (1-\rho)\,D^-$, and
$\ARDone$ gives $D^+ = D^{\mathrm{last}} + (1-\rho)\,(D^- - D^{\mathrm{last}})$.
After a repair the damage is, in general, not a Gamma variable: it is a weighted
sum of independent Gamma terms and known constants. The reliability row below
holds for every such sum.
```

### M-B-26: Gamma: the day map is exact for the true mean

- From: Gamma Model, eq:gm_map, lines 476-478 of the frozen copy.
- Kind: remark.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: the day map and the rows table.

```latex
with $\delta\mu_k = \sum_{j=1}^{M} x_{ijk}\,\mu^{\inc}_{jk}$. The map is exact
for the true mean $\mathbb{E}D_k$, because expectation is linear. The rows are
the generic rows of Section~\ref{sec:encoding}, with the data below.
```

### M-B-27: Gamma: smallest rate, with the units remark

- From: Gamma Model, eq:gm_betastar, lines 496-504 of the frozen copy.
- Kind: remark.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: the rule for beta* and the row for a cell without a random term.

```latex
where $j$ runs over the missions and $k$ over the days of both phases, and
$\beta_0$, $\beta^{\new}$ enter only when they are given; $\beta^{\new}$ enters
only when replacements are allowed (\texttt{allow\_replacement}). So $\beta^*$ is the
smallest rate among the terms with a positive mean: a mission with
$\mu^{\inc}_{jk} = 0$ adds no term, and an omitted rate declares a known damage,
which is a constant and does not enter $\beta^*$. If no term has a rate, the
damage is a known constant, and the row below is $\mu_k \leq \tau$. Write
$t = \beta^*\tau$. Units of $1/\beta^*$ turn every law below into a law of rate
at least 1.
```

### M-B-28: Gamma: class of true states (definition, lemma, proof)

- From: Gamma Model, def:gm_class, lem:gm_class, lines 506-550 of the frozen copy.
- Kind: statement and proof.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\subsubsection{Class of true states}

\begin{definition}[Class of weighted Gamma sums]
\label{def:gm_class}
Let $m \geq 0$. The class $\mathcal{G}(m)$ holds the laws of the random
variables
\[
  S = \delta + \sum_{i=1}^{n} c_i\, Y_i, \qquad
  Y_i \sim \mathrm{Gamma}(a_i, 1) \text{ independent}, \quad
  c_i \in (0, 1], \quad a_i > 0, \quad \delta \geq 0,
\]
with $n \geq 0$ and mean $\delta + \sum_i c_i a_i = m$. The term $c\,Y$ with
$Y \sim \mathrm{Gamma}(m'/c, 1)$ has mean $m'$; as $c \to 0$ it tends to the
constant $m'$.
\end{definition}

\begin{lemma}[True states belong to the class]
\label{lem:gm_class}
Let $D_k$ be the damage of a Gamma cell on day $k$ of a schedule, under either
law. Then the law of $\beta^* D_k$ belongs to $\mathcal{G}(\beta^*\,\mathbb{E}D_k)$.
\end{lemma}

\begin{proof}
Since the last replacement (or day 0), the damage is
$D_k = \delta + \sum_t w_t X_t$: the $X_t$ are the independent Gamma terms added
since then, $\delta \geq 0$ is the weighted sum of the known constants, and each
weight $w_t \in [0, 1]$ is a product of factors $1-\rho$; a term of weight 0
(possible when $\rho = 1$) is dropped. A mission adds a term
with weight 1. $\ARDinf$ multiplies every weight, constants included, by
$1-\rho$. $\ARDone$ multiplies by $1-\rho$ the weights of the terms and constants
added since the last intervention, and keeps the others. A replacement starts a
new sum with the replacement draw. When the starting anchor $g^{\mu}_0$ is below
$\mu_0$, the initial damage is read as two independent terms of rate $\beta_0$,
with means $g^{\mu}_0$ and $\mu_0 - g^{\mu}_0$, whose sum has the law
$\mathrm{Gamma}(\beta_0\mu_0, \beta_0)$; a known initial damage is read as the
two constants $g^{\mu}_0$ and $\mu_0 - g^{\mu}_0$. Write $X_t = Y_t/\beta_t$ with
$Y_t \sim \mathrm{Gamma}(a_t, 1)$ and $\beta_t \geq \beta^*$ by
Eq.~\eqref{eq:gm_betastar}. Then
\[
  \beta^* D_k = \beta^*\delta + \sum_t \frac{w_t\,\beta^*}{\beta_t}\, Y_t,
  \qquad \frac{w_t\,\beta^*}{\beta_t} \in (0, 1].
\]
The $Y_t$ are independent by Assumption G1. Therefore the law of $\beta^* D_k$
is in $\mathcal{G}(\beta^*\,\mathbb{E}D_k)$.
\end{proof}
```

### M-B-29: Gamma: the largest tail at a given mean (W, rates lemma, one-term theorem and proof)

- From: Gamma Model, lem:gm_rates, thm:gm_oneterm, lines 552-665 of the frozen copy.
- Kind: statement and proof.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: the definition of Q(a, x).

```latex
\subsubsection{The largest tail at a given mean}

For $0 < m < t$, let $W(m, t)$ be the supremum of $\Pr(S > t)$ over the random
variables $S$ whose law is in $\mathcal{G}(m)$.
$Q(a, x) = \Pr(\mathrm{Gamma}(a, 1) > x)$ is the regularised upper incomplete
gamma function (\texttt{scipy.special.gammaincc(a, x)}). The one-term member
$c\,\mathrm{Gamma}(m/c, 1)$ has the tail $Q(m/c,\, t/c)$ at $t$.

A member of $\mathcal{G}(m)$ is described by its atoms: the pairs
$(m_i, c_i)$ of a term mean $m_i = c_i a_i$ and a scale $c_i$, and the constant
$\delta$, read as an atom at scale $c = 0$. Terms with equal scales merge into
one term, because a sum of independent Gamma variables of one rate is a Gamma
variable.

\begin{lemma}[First-order rates]
\label{lem:gm_rates}
Let $S$ be a member with at least one term, with density $f_S$. Since
$\delta \leq m < t$, $f_S$ is continuous at $t$. For $c \in (0, 1]$ let
\[
  \varphi_S(c) = \frac{1}{c}\int_0^\infty \Pr(t - y < S \leq t)\,
    \frac{e^{-y/c}}{y}\, dy,
  \qquad \varphi_S(0) = f_S(t).
\]
(a) Adding mean $\Delta m$ at scale $c$ (a term $c\,\mathrm{Gamma}(\Delta m/c, 1)$,
or the constant $\Delta m$ when $c = 0$) raises $\Pr(S > t)$ at the rate
$\varphi_S(c)$ as $\Delta m \to 0$; removing mean from an atom at scale $c$ lowers
it at the same rate. (b) Moving the scale of an atom $(m_i, c_i)$,
$c_i \in (0,1)$, at fixed mean $m_i$ changes $\Pr(S > t)$ at the rate
$m_i\, \partial\varphi_S/\partial c\,(c_i)$.
\end{lemma}

\begin{proof}
$S$ is infinitely divisible, with drift $\delta$ and Lévy density
$\ell(y) = \sum_i (m_i/c_i)\, e^{-y/c_i}/y$ on $y > 0$. A change of the
parameters at the rates $(\dot\delta, \dot\ell)$ changes the Laplace transform
$\mathbb{E}e^{-\sigma S} = \exp\bigl(-\sigma\delta - \int_0^\infty (1 - e^{-\sigma y})\,
\ell(y)\, dy\bigr)$ at the rate
$\int_0^\infty \mathbb{E}\bigl[e^{-\sigma(S+y)} - e^{-\sigma S}\bigr]\,\dot\ell(y)\, dy
- \sigma\dot\delta\,\mathbb{E}e^{-\sigma S}$. This is the transform of the rate
$\int_0^\infty \Pr(t - y < S \leq t)\,\dot\ell(y)\, dy + \dot\delta\, f_S(t)$ of
$\Pr(S > t)$. Case (a) is $\dot\ell(y) = c^{-1}e^{-y/c}/y$, or $\dot\delta = 1$
when $c = 0$. Case (b) is
$\dot\ell(y) = m_i\,\partial_c\bigl(c^{-1}e^{-y/c}/y\bigr)$ at $c = c_i$.
\end{proof}

\begin{theorem}[One term is the worst case]
\label{thm:gm_oneterm}
Let $0 < m < t$. Then
\begin{equation}
  W(m, t) = \max_{c \in (0, 1]} Q\!\left(\frac{m}{c},\, \frac{t}{c}\right).
  \label{eq:gm_oneterm}
\end{equation}
\end{theorem}

\begin{proof}
Let $\mathcal{G}_N(m)$ be the members with at most $N$ atoms, the constant
included, and let $N \geq 3$.
\begin{enumerate}[leftmargin=1.5em]
  \item \emph{A maximiser exists.} $\mathcal{G}_N(m)$ is the image of the
        compact set of atoms $(m_i, c_i)$ with $m_i \geq 0$, $\sum_i m_i = m$,
        $c_i \in [0, 1]$. The law of $S$ depends continuously on the atoms, and
        only the constant member $S = m$ has an atom, at $m \neq t$. So
        $\Pr(S > t)$ is continuous and attains its maximum at some $S^*$. Since
        $\Pr(m > t) = 0 < Q(m, t)$, $S^*$ has at least one term. Every member
        has finitely many atoms, so
        $W(m, t) = \sup_N \max_{\mathcal{G}_N(m)} \Pr(S > t)$.
  \item \emph{First-order conditions.} Write $\varphi^* = \varphi_{S^*}$. By
        Lemma~\ref{lem:gm_rates}: (i) moving mean between two atoms of positive
        mean must not raise the tail, so $\varphi^*$ takes one value $\nu^*$
        at every atom of positive mean; (ii) at an atom with
        $c \in (0, 1)$, $\partial\varphi^*/\partial c = 0$; (iii) if $S^*$ has
        fewer than $N$ atoms of positive mean, a new atom can be opened at any
        $c \in [0, 1]$, so $\varphi^*(c) \leq \nu^*$ on $[0, 1]$.
  \item \emph{A single-peak function.} Put $\lambda = 1/c$ and
        $h(u) = \Pr(t - u < S^* \leq t)/u$. Then
        $\varphi^*(\lambda) = \lambda\int_0^\infty h(u)\, e^{-\lambda u}\, du$,
        and $\varphi^*(\infty) = h(0^+) = f_{S^*}(t)$ is the value at the
        constant atom. $S^*$ is a constant plus a sum of independent Gamma
        variables, so its density is unimodal
        \cite[Theorem~4, p.~11]{roosta2015}, proved in
        \cite[proof of Theorem~4, pp.~194--195]{szekely2003}; the constant
        shifts the mode only. Then $g(v) = f_{S^*}(t - v)$ for $v < t$ and
        $g(v) = 0$ for $v \geq t$ is unimodal on $[0, \infty)$, and so is its
        running average $h(u) = u^{-1}\int_0^u g$: before the mode of $g$,
        $g \geq h$; after it, $u\,(g(u) - h(u)) = u\,g(u) - \int_0^u g$ is
        nonincreasing. Since $h' = (g - h)/u$, $h$ rises and then falls.
  \item \emph{Zero count.} For every constant $\nu'$, $h - \nu'$ changes
        sign at most twice. The Laplace transform of a function with at most
        $n$ sign changes has at most $n$ zeros in $\lambda > 0$, counted with
        multiplicity: at a sign change $u_1$, multiply the transform by
        $e^{\lambda u_1}$; its derivative in $\lambda$ is the transform of a
        function with one sign change fewer; Rolle's theorem and induction
        conclude. So $\varphi^* - \nu' = \lambda\int_0^\infty (h - \nu')\,
        e^{-\lambda u}\, du$ has at most two zeros in $\lambda > 0$. Also
        $\varphi^*(\lambda) \to 0$ as $\lambda \to 0^+$. Hence $\varphi^*$
        cannot rise above a level $\nu' > 0$, fall below it and rise above it
        again: with $\varphi^*(0^+) = 0$ that would give three zeros. So
        $\varphi^*$ is quasi-concave. It is analytic in $\lambda > 0$ and not
        constant, so its maximum over $\lambda \in [1, \infty]$ is attained at
        one point only: two maximisers would make $\varphi^*$ constant between
        them.
  \item \emph{One term.} By (i) and (ii), an atom with $c \in (0, 1)$ is a zero
        of $\varphi^* - \nu^*$ of multiplicity at least 2, and an atom at
        $c = 1$ a zero of multiplicity at least 1; the constant atom is at
        $\lambda = \infty$. By step~4, $S^*$ has at most two atoms of positive
        mean. As $N \geq 3$, (iii) applies, so $\nu^*$ is the maximum of
        $\varphi^*$ over $[1, \infty]$, attained at every atom of positive mean.
        By step~4 there is only one such atom, and by step~1 it is not the
        constant. So $S^* = c^*\,\mathrm{Gamma}(m/c^*, 1)$ with
        $c^* \in (0, 1]$.
\end{enumerate}
Therefore $\max_{\mathcal{G}_N(m)} \Pr(S > t) = \max_{c \in (0,1]} Q(m/c, t/c)$
for every $N \geq 3$, and Eq.~\eqref{eq:gm_oneterm} follows from step~1.
\end{proof}
```

### M-B-30: Gamma: W is monotone in the mean

- From: Gamma Model, lem:gm_monotone, lines 667-676 of the frozen copy.
- Kind: statement and proof.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\begin{lemma}[Monotone in the mean]
\label{lem:gm_monotone}
$W(m, t)$ is nondecreasing in $m \in (0, t)$.
\end{lemma}

\begin{proof}
Let $m < m' < t$ and let the law of $S$ be in $\mathcal{G}(m)$. Then the law of
$S + (m' - m)$ is in $\mathcal{G}(m')$, and
$\Pr(S + m' - m > t) \geq \Pr(S > t)$. Therefore $W(m, t) \leq W(m', t)$.
\end{proof}
```

### M-B-31: Gamma: the cap paragraph with the reasons (w < 1/2, limit 1/2)

- From: Gamma Model, Reliability row, The cap, lines 680-697 of the frozen copy.
- Kind: derivation.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: the definitions of w and A_1, the rule for eps >= 1/2, the row eq:gm_row.

```latex
\textbf{The cap.} For $A \in [0, t)$ let
$w(A) = \max_{c \in (0, 1]} Q(A/c,\, t/c)$; it is continuous and nondecreasing
in $A$, because $Q(a, x)$ increases in $a$. Also $w(A) < 1/2$ for every
$A < t$: with $r = A/t < 1$ and $s = t/c$,
$Q(rs, s) \leq Q(rs, rs) < 1/2$, because the median of a Gamma law lies below its
mean; and $w(A) \to 1/2$ as $A \to t$, because $Q(s, s) \to 1/2$ as
$s \to \infty$. The \emph{sharp cap} $A_1(t, \varepsilon)$ is the largest
$A \in [0, t)$ with $w(A) \leq \varepsilon$; for $\varepsilon < 1/2$ it exists
and lies below $t$. For $\varepsilon \geq 1/2$ no largest value exists, because
every $A < t$ qualifies; $A_1$ is then the value that step~3 of the computation
below returns, which lies below $t$. The row, on every day
$k = 1, \ldots, T$, is
\begin{equation}
  \mu_k \;\leq\; \frac{A_1(\beta^*\tau,\, \varepsilon)}{\beta^*}.
  \label{eq:gm_row}
\end{equation}
The row is linear: it is built in its exact form, and a value of
\texttt{reliability\_impl} at the cell is ignored (Section~\ref{sec:linear_forms}).
```

### M-B-32: Gamma: the row is safe (corollary and proof)

- From: Gamma Model, cor:gm_safe, lines 699-716 of the frozen copy.
- Kind: statement and proof.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\begin{corollary}[The row is safe]
\label{cor:gm_safe}
Let the tracked mean satisfy Eq.~\eqref{eq:gm_row} on day $k$. Then
$\Pr(D_k > \tau) \leq \varepsilon$.
\end{corollary}

\begin{proof}
Let $m = \beta^*\,\mathbb{E}D_k$ and $A = A_1(\beta^*\tau, \varepsilon)$. The
tracked mean bounds the true mean from above (Section~\ref{sec:encoding}), so
$m \leq \beta^*\mu_k \leq A < t$. If $m = 0$, then $D_k = 0$. Otherwise
\[
  \Pr(D_k > \tau) = \Pr(\beta^* D_k > t)
  \leq W(m, t) \leq W(A, t) = w(A) \leq \varepsilon,
\]
by Lemma~\ref{lem:gm_class}, Lemma~\ref{lem:gm_monotone},
Theorem~\ref{thm:gm_oneterm} and the choice of $A$: in both cases of the
definition, $A < t$ and $w(A) \leq \varepsilon$.
\end{proof}
```

### M-B-33: Gamma: computation of A_1 with its justifications

- From: Gamma Model, Computation of A_1, lines 718-737 of the frozen copy.
- Kind: rationale.
- Suggested destination: Implementation details.
- Spec keeps: the three steps, without the justifications (single peak checked numerically; Markov bound).

```latex
\textbf{Computation of $A_1$} (offline, once per cell, before the model is built):
\begin{enumerate}[leftmargin=1.5em, nosep]
  \item $t = \beta^*\tau$. $\hat A$ is the root of $Q(\hat A, t) = \varepsilon$,
        found by bisection on $\hat A$ ($Q$ increases in its first argument;
        \texttt{gammaincinv} solves for the second argument and does not apply).
  \item $w(A)$: the function $c \mapsto Q(A/c, t/c)$ tends to 0 as $c \to 0$
        and has a single peak on $(0, 1]$ (checked numerically for $t$ from
        $10^{-3}$ to $100$ and $A/t$ from $0.05$ to $0.999$). Its maximum is found on a
        logarithmic grid of at least 400 values of $c$ in
        $[\min(10^{-4}, t/100),\, 1]$, then refined by a golden-section search
        between the neighbours of the best grid point. If the best grid point is
        the smallest value of $c$, the grid is extended downwards.
  \item If $\hat A < t$ and $w(\hat A) \leq \varepsilon$, then
        $A_1 = \hat A$. Otherwise $A_1$ is found by bisection on $A$ in
        $[\varepsilon t,\, \min(\hat A, t))$, to a relative tolerance of
        $10^{-9}$. The lower end satisfies $w(\varepsilon t) \leq \varepsilon$,
        by Markov's inequality $Q(a, x) \leq a/x$. The bisection returns the
        lower end of its last interval, so $w(A_1) \leq \varepsilon$ holds at
        the returned value.
\end{enumerate}
```

### M-B-34: Gamma: the sharp cap and the exact one-rate cap (t1, table, witness)

- From: Gamma Model, remark, lines 739-761 of the frozen copy.
- Kind: example.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\emph{Remark (the sharp cap and the exact one-rate cap).} $\hat A$ is the exact
cap of a single Gamma law of rate $\beta^*$, and $A_1 \leq \hat A$, because
$c = 1$ is in the maximum. Let $t_1(\varepsilon)$ be the value of $t$ at which
$\partial_c Q(\hat A/c, t/c)$ at $c = 1$ changes sign: $t_1 = 0.436$, $0.448$
and $0.614$ for $\varepsilon = 0.001$, $0.01$ and $0.1$. For $t \geq t_1$,
$A_1 = \hat A$: on a grid of $t$ in $[t_1, 1000]$, no $c < 1$ gives a larger tail
than $c = 1$. Below $t_1$, $A_1 < \hat A$. For $\varepsilon = 0.01$:
\begin{center}
\begin{tabular}{@{}lccccc@{}}
\toprule
$t = \beta^*\tau$ & 0.01 & 0.1 & 0.3 & 1 & 10 \\
\midrule
$A_1/\hat A$ & 0.142 & 0.642 & 0.961 & 1 & 1 \\
\bottomrule
\end{tabular}
\end{center}
When every term has the same rate and no repair occurs, the true state is a single
Gamma law and $\hat A$ is exact for it. With several rates, or with repairs, the true state is a
weighted sum, and below $t_1$ the exact one-rate cap is not safe. Example:
$\varepsilon = 0.01$ and $\beta^*\tau = 0.1$, so $\hat A = 0.0055012$. A new
part flies one mission of mean $\hat A/\beta^*$ and rate $4.56\,\beta^*$. Its
mean meets the exact one-rate cap, but its tail is $1.565\,\varepsilon$. The
sharp cap is $A_1 = 0.642\,\hat A$ there.
```

### M-B-35: Gamma: below the threshold, the fixed fraction r*

- From: Gamma Model, remark, lines 763-770 of the frozen copy.
- Kind: remark.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\emph{Remark (below the threshold).} With $r = A/t$ and $s = t/c$,
$Q(A/c, t/c) = Q(r s, s)$, and $c \in (0, 1]$ means $s \geq t$. For
$t \leq t_1(\varepsilon)$ the peak of $s \mapsto Q(rs, s)$ at the cap lies at
$s \geq t$, so the maximum does not depend on $t$, and
$A_1(t, \varepsilon) = r^*(\varepsilon)\, t$. The mean cap is then
$A_1/\beta^* = r^*(\varepsilon)\,\tau$, whatever the rates:
$r^* = 0.00355$, $0.0353$ and $0.331$ for $\varepsilon = 0.001$, $0.01$ and
$0.1$.
```

### M-B-36: Gamma: the choice of beta*

- From: Gamma Model, remark, lines 772-777 of the frozen copy.
- Kind: rationale.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\emph{Remark (the choice of $\beta^*$).} A larger $t$ shrinks the range
$s \geq t$, so $A_1(t, \varepsilon)/t$ is nondecreasing in $t$, and the mean cap
$A_1(\beta^*\tau, \varepsilon)/\beta^* = \tau\, A_1(t, \varepsilon)/t$ is
nondecreasing in $\beta^*$. Lemma~\ref{lem:gm_class} holds for every
$\beta^*$ at most the smallest rate. So the smallest rate gives the largest safe
cap of this form.
```

### M-B-37: Gamma: conservatism of the class

- From: Gamma Model, remark, lines 779-782 of the frozen copy.
- Kind: remark.
- Suggested destination: Stochastic orderings: within each degradation family.
- Spec keeps: nothing.

```latex
\emph{Remark (conservatism).} The class $\mathcal{G}(m)$ also contains laws that
no schedule produces. Below $t_1$ the row can therefore reject some safe
schedules. The row keeps each mission's own mean; $\beta^*$ only bounds how
dispersed each term may be.
```

### M-B-38: Gamma: facts (M1)-(M3)

- From: Gamma Model, lines 784-792 of the frozen copy.
- Kind: statement.
- Suggested destination: Stochastic orderings: loop constraint for each degradation model.
- Spec keeps: nothing.

```latex
\textbf{Facts (M1)--(M3) of Section~\ref{sec:encoding}.}
(M1) The repair factor $1-\rho$ lies in $[0, 1]$, and $\delta\mu_k \geq 0$.
(M2) $\theta_0 \leq \xi_0$, as in Section~\ref{sec:dyn_gaussian}; on every later
day the true mean equals the map~\eqref{eq:gm_map} applied to the true statistics
of the day before.
(M3) If a value $\mu \leq A_1/\beta^*$ is at least the true mean of day $k$, then
$\Pr(D_k > \tau) \leq \varepsilon$: the proof of Corollary~\ref{cor:gm_safe} uses
only these two facts. The loop row of Section~\ref{sec:loop} covers $\mu$ and,
under $\ARDone$, $g^{\mu}$.
```

### M-B-39: Rainflow: distribution-free model, introduction

- From: Rainflow Model (sec:dyn_rainflow), lines 803-811 of the frozen copy.
- Kind: remark.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: a two-line statement of the inputs and the five bounds.

```latex
The rainflow model is \emph{distribution-free}. The law of a mission increment
is measured from recorded load histories by rainflow counting (Palmgren--Miner
damage), and it enters the model only through summary statistics: its mean, and,
as the bound requires, its variance, the width of its range, or its cumulant
generating function (CGF) at a fixed point $s$. No parametric family is assumed.
The reliability requirement $\Pr(D_k > \tau) \leq \varepsilon$ is certified by
one of five tail bounds, chosen per cell by \texttt{bound\_method}:
\texttt{markov}, \texttt{cantelli}, \texttt{hoeffding}, \texttt{bernstein},
\texttt{chernoff}.
```

### M-B-40: Rainflow: modelling assumptions A0-A3, full statements

- From: Rainflow Model, Modelling assumptions, lines 813-833 of the frozen copy.
- Kind: statement.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: A0-A3 as one-line requirements on the input data.

```latex
\textbf{Modelling assumptions (normative).}
\begin{itemize}[leftmargin=1.5em]
  \item \textbf{A0 (non-negative damage).} Every increment, the initial damage
        and every replacement draw are non-negative.
  \item \textbf{A1 (independence).} The mission increments are mutually
        independent, and independent of the initial damage and of every
        replacement draw. Each replacement installs a new draw, independent of
        everything else. Every bound except Markov needs A1.
  \item \textbf{A2 (bounded ranges; Hoeffding and Bernstein).} Each mission
        increment lies in an interval of width $b^{\inc}_{jk}$ (input
        \texttt{support}). The initial damage is a known part plus independent
        terms, each in an interval; $R_0$ is the sum of their squared widths
        (input \texttt{R\_0}) and $b^0$ the largest width (input
        \texttt{support\_0}). The replacement draw is described in the same way by
        $R^{\new}$ and $b^{\new}$ (inputs \texttt{R\_new}, \texttt{support\_new}).
        A known damage has $R = b = 0$, the defaults.
  \item \textbf{A3 (repetition).} The row of the cell's bound holds on every day
        $k = 1, \ldots, T$, and after day $T$ the operating phase repeats with
        period $H_2$, schedule and increment profile included. This is the
        setting of the loop argument of Section~\ref{sec:loop}.
\end{itemize}
```

### M-B-41: Rainflow: width of a range (use by the bounds, Popoviciu)

- From: Rainflow Model, Width of a range, lines 835-842 of the frozen copy.
- Kind: rationale.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: the definition of the width and the check v <= b^2/4 by reference.

```latex
\textbf{Width of a range.} The input \texttt{support} of an increment $X$ is the
width $b$ of an interval $[l, l + b]$ that contains $X$ almost surely. Hoeffding's
inequality uses the widths. Bernstein's inequality needs an upper bound of the
centred term, and the width is one: $X - \mathbb{E}X \leq l + b - l = b$, since
$\mathbb{E}X \geq l$. A width $b$ is consistent with a variance $v$ only if
$v \leq b^2/4$ (Popoviciu's inequality:
$\mathrm{Var}\,X \leq \mathbb{E}(X - l - b/2)^2 \leq (b/2)^2$); the check is in
Section~\ref{sec:consistency_checks}.
```

### M-B-42: Rainflow: day maps of the statistics (term representation, proposition, proof)

- From: Rainflow Model, Day maps, prop:rf_maps, lines 882-940 of the frozen copy.
- Kind: statement and proof.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: each state follows the generic map with the factor a of the rows table; anchor = value after the last intervention.

```latex
\subsubsection{Day maps}

Between replacements the damage is $D_k = \sum_t w_t X_t$: the $X_t$ are the
known part, the independent terms of the initial damage or of the replacement
draw, and the mission increments; each weight $w_t \in [0, 1]$ is a product of
factors $1 - \rho$, and a term of weight 0 (possible when $\rho = 1$) is
dropped. Under $\ARDone$, write $D^- = G + \Delta$, with $G$ the damage
just after the last intervention and $\Delta$ the weighted increments added
since. $G$ and $\Delta$ are independent by A1, and a repair gives
$D^+ = G + (1-\rho)\,\Delta$. Before the first intervention, $G$ is the part of
the initial damage that the starting anchors describe
(Section~\ref{sec:repair_laws}).

For a damage $D = \sum_t w_t X_t$, write $m(D) = \mathbb{E}D$,
$V(D) = \mathrm{Var}\,D$, $R(D) = \sum_t w_t^2 r_t$ with $r_t$ the squared width
of $X_t$, and $K(D) = \ln\mathbb{E}\,e^{sD}$.

\begin{proposition}[Day maps of the statistics]
\label{prop:rf_maps}
Let $\theta \in \{m, V, R\}$ with factor $a_\theta = 1-\rho$ for $m$ and
$a_\theta = (1-\rho)^2$ for $V$ and $R$. Under A1:
\begin{enumerate}[leftmargin=1.5em, nosep]
  \item a mission day adds the increment's statistic:
        $\theta(D + X) = \theta(D) + \theta(X)$ and $K(D + X) = K(D) + K(X)$;
  \item $\ARDinf$: $\theta\bigl((1-\rho)D\bigr) = a_\theta\,\theta(D)$ and
        $K\bigl((1-\rho)D\bigr) \leq (1-\rho)\,K(D)$;
  \item $\ARDone$: $\theta(D^+) = a_\theta\,\theta(D^-) + (1 - a_\theta)\,\theta(G)$
        and $K(D^+) \leq (1-\rho)\,K(D^-) + \rho\,K(G)$.
\end{enumerate}
\end{proposition}

\begin{proof}
Item 1: the mean and $R$ are additive by definition, the variance and the CGF by
independence. Item 2: $m$ and $V$ scale by $1-\rho$ and $(1-\rho)^2$; every
weight is multiplied by $1-\rho$, so $R$ scales by $(1-\rho)^2$. A CGF is convex
with value 0 at 0, so
$K\bigl((1-\rho)D\bigr) = K_D\bigl((1-\rho)s\bigr) \leq (1-\rho)\,K_D(s) + \rho\,K_D(0)
= (1-\rho)\,K(D)$, with $K_D$ the CGF of $D$ as a function of its argument.
Item 3: by independence, $\theta(D^-) = \theta(G) + \theta(\Delta)$; for $R$ the
weights of the terms of $G$ are unchanged and those of $\Delta$ are multiplied by
$1-\rho$. Hence
\[
  \theta(D^+) = \theta(G) + a_\theta\,\theta(\Delta)
  = \theta(G) + a_\theta\bigl(\theta(D^-) - \theta(G)\bigr)
  = a_\theta\,\theta(D^-) + (1 - a_\theta)\,\theta(G),
\]
and, with item 2 applied to $\Delta$,
\[
  K(D^+) = K(G) + K_\Delta\bigl((1-\rho)s\bigr)
  \leq K(G) + (1-\rho)\bigl(K(D^-) - K(G)\bigr)
  = (1-\rho)\,K(D^-) + \rho\,K(G).
\]
\end{proof}

So each tracked state follows the generic map of Section~\ref{sec:encoding} with
the factor $a = 1-\rho$ for $\mu$ and $K$, and $a = (1-\rho)^2$ for $v$ and $R$.
Its anchor is its value just after the last intervention. The maps of $\mu$,
$v$ and $R$ are exact; the map of $K$ is an upper bound, which is what the
domination argument of Section~\ref{sec:encoding} needs.
```

### M-B-43: Rainflow: why the starting and replacement defaults of R and K (Jensen, example)

- From: Rainflow Model, Starting and replacement values, lines 954-968 of the frozen copy.
- Kind: rationale.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: one sentence: the defaults describe a known damage; a random one needs its inputs (pointer).

```latex
\emph{Why the defaults.} By Jensen's inequality,
$K(D) = \ln\mathbb{E}\,e^{sD} \geq s\,\mathbb{E}D$, with equality exactly when $D$
is almost surely constant. So $s\mu_0$ is the exact CGF of a known initial
damage $\mu_0$, and a lower bound for a random one; a random initial damage needs
its own CGF in \texttt{cgf\_0}. The input checks of
Section~\ref{sec:consistency_checks} require $\texttt{cgf} \geq s\,\mu^{\inc}$ for
every increment, $K_0 \geq s\mu_0$ and $K^{\new} \geq s\mu^{\new}$. A known damage
has width 0, so $R_0 = 0$ is exact for it, and a random one needs \texttt{R\_0}.
The same holds for the replacement draw. A starting value of $K$ below $s\mu_0$ makes the row optimistic. Example:
$\tau = 1$, $\varepsilon = 0.1$, $s = 5$, a known initial damage $\mu_0 = 0.6$,
and one mission that adds 0 or 0.5 with probability $1/2$ each, so
$K^{\inc} = \ln(0.5 + 0.5\,e^{2.5}) = 1.886$. Then
$\Pr(D > \tau) = 0.5$. The row with $K_0 = s\mu_0 = 3$ gives
$4.886 - 5 > \ln 0.1$ and rejects the plan; a start at $K_0 = 0$ would give
$1.886 - 5 = -3.11 \leq \ln 0.1 = -2.30$ and accept it.
```

### M-B-44: Rainflow: true statistics below the tracked states

- From: Rainflow Model, Reliability rows, lines 1001-1004 of the frozen copy.
- Kind: derivation.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: nothing.

```latex
Write $d_k = \tau - \mu_k$. Let $(m, V, R^{\mathrm{t}}, K^{\mathrm{t}})$ be the
true statistics of $D_k$. By Proposition~\ref{prop:rf_maps} and
Section~\ref{sec:encoding}, each is at most the tracked state of the same name.
Each row below is imposed on every day $k = 1, \ldots, T$.
```

### M-B-45: Rainflow: Markov row, derivation

- From: Rainflow Model, eq:rf_markov, lines 1006-1012 of the frozen copy.
- Kind: derivation.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: the row eq:rf_markov.

```latex
\textbf{Markov} (\texttt{markov}). By A0 and Markov's inequality,
$\Pr(D_k > \tau) \leq m/\tau \leq \mu_k/\tau$. The row is
\begin{equation}
  \mu_k \leq \varepsilon\,\tau.
  \label{eq:rf_markov}
\end{equation}
It uses the mean only and does not need A1 or A2.
```

### M-B-46: Rainflow: Cantelli row, derivation and validity

- From: Rainflow Model, eq:rf_cantelli, lines 1014-1027 of the frozen copy.
- Kind: derivation.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: the row eq:rf_cantelli and kappa_C.

```latex
\textbf{Cantelli} (\texttt{cantelli}). Cantelli's inequality gives
$\Pr(D - m \geq a) \leq V/(V + a^2)$ for $a > 0$. The row is
\begin{equation}
  (1 - \varepsilon)\, v_k \leq \varepsilon\,(\tau - \mu_k)^2, \qquad \mu_k \leq \tau,
  \label{eq:rf_cantelli}
\end{equation}
equivalently $\mu_k + \kappa_C\sqrt{v_k} \leq \tau$ with
$\kappa_C = \sqrt{(1-\varepsilon)/\varepsilon}$. \emph{Validity.} If $d_k > 0$,
\[
  \Pr(D_k > \tau) \leq \Pr(D_k - m \geq \tau - m) \leq \frac{V}{V + (\tau - m)^2}
  \leq \frac{v_k}{v_k + d_k^2} \leq \varepsilon,
\]
because $v/(v + d^2)$ increases in $v$ and decreases in $d$, and $\tau - m \geq d_k$.
If $d_k = 0$, the row gives $v_k = 0$, so $V = 0$ and $D_k = m \leq \tau$.
```

### M-B-47: Rainflow: Hoeffding row, derivation and validity

- From: Rainflow Model, eq:rf_hoeffding, lines 1029-1044 of the frozen copy.
- Kind: derivation.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: the row eq:rf_hoeffding.

```latex
\textbf{Hoeffding} (\texttt{hoeffding}). $D_k - m$ is a sum of independent
centred terms $w_t(X_t - \mathbb{E}X_t)$ with ranges of widths $w_t\sqrt{r_t}$,
and the sum of their squares is $R^{\mathrm{t}}$. Hoeffding's inequality gives
$\Pr(D_k - m \geq a) \leq \exp(-2a^2/R^{\mathrm{t}})$, and
$\exp(-2d^2/R) \leq \varepsilon$ exactly when $d^2 \geq (\kappa/2)\,R$. The row
is
\begin{equation}
  (\tau - \mu_k)^2 \geq \frac{\kappa}{2}\, R_k, \qquad \mu_k \leq \tau.
  \label{eq:rf_hoeffding}
\end{equation}
\emph{Validity.} If $R^{\mathrm{t}} = 0$, every term is constant and
$D_k = m \leq \tau$; this covers $d_k = 0$, where the row gives $R_k = 0$.
Otherwise $d_k > 0$, and
$\Pr(D_k > \tau) \leq \exp\bigl(-2(\tau - m)^2/R^{\mathrm{t}}\bigr)
\leq \exp(-2d_k^2/R_k) \leq \varepsilon$, because $\exp(-2d^2/R)$ increases in
$R$ and decreases in $d$, and $\tau - m \geq d_k$.
```

### M-B-48: Rainflow: Bernstein support constant and row, derivation and validity

- From: Rainflow Model, eq:rf_bsupport, eq:rf_bernstein, lines 1046-1069 of the frozen copy.
- Kind: derivation.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: b, c (eq:rf_bsupport), the row eq:rf_bernstein with its side row.

```latex
\textbf{Bernstein} (\texttt{bernstein}). The \emph{support constant} of the cell
is
\begin{equation}
  b = \max\Bigl(b^0,\; b^{\new},\; \max_{j,k} b^{\inc}_{jk}\Bigr), \qquad
  c = \frac{\kappa\, b}{3},
  \label{eq:rf_bsupport}
\end{equation}
over the missions $j$ and the days $k$ of both phases; $b^{\new}$ enters only
when replacements are allowed (\texttt{allow\_replacement}). It is a constant, not
a decision variable. Every weighted term satisfies
$w_t(X_t - \mathbb{E}X_t) \leq w_t\, b_t \leq b$, whatever the schedule, because
$w_t \leq 1$. Bernstein's inequality for independent terms bounded above by $b$
gives $\Pr(D - m \geq a) \leq \exp\bigl(-a^2/(2(V + b\,a/3))\bigr)$ for $a > 0$;
the right side is at most $\varepsilon$ exactly when
$a^2 - 2c\,a - 2\kappa V \geq 0$. The row is
\begin{equation}
  (\tau - c - \mu_k)^2 \geq c^2 + 2\kappa\, v_k, \qquad \mu_k \leq \tau - c,
  \label{eq:rf_bernstein}
\end{equation}
equivalently $\mu_k + c + \sqrt{c^2 + 2\kappa v_k} \leq \tau$. With $v_k \geq 0$,
the two rows give $d_k \geq 2c$.
\emph{Validity.} $a^2/(2(V + b\,a/3))$ increases in $a > 0$ and decreases in $V$,
and $\tau - m \geq d_k \geq 2c$; so $\Pr(D_k > \tau) \leq \varepsilon$ when
$b > 0$. When $b = 0$, every term is constant and $D_k = m \leq \tau$.
```

### M-B-49: Rainflow: Bernstein side row and precondition

- From: Rainflow Model, remark, lines 1071-1077 of the frozen copy.
- Kind: rationale.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: the side row in every form; tau > 2 kappa b/3 as a requirement.

```latex
\emph{Remark (the side row).} The quadratic row alone also admits the point
$d_k = 0$, $v_k = 0$: a known damage equal to $\tau$. That point is safe, but it is
cut off from the rest of the feasible set. The side row
$\mu_k \leq \tau - c$ removes it and keeps the branch
$d_k \geq c + \sqrt{c^2 + 2\kappa v_k}$. At $\mu = v = 0$ the row needs
$\tau \geq 2c$; a positive variance needs $\tau > 2\kappa b/3$, the input check
of Section~\ref{sec:consistency_checks}.
```

### M-B-50: Rainflow: Chernoff row, derivation and validity

- From: Rainflow Model, eq:rf_chernoff, lines 1079-1093 of the frozen copy.
- Kind: derivation.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution.
- Spec keeps: the row eq:rf_chernoff, the range s tau - kappa, the requirements s > kappa/tau and K_new <= s tau - kappa.

```latex
\textbf{Chernoff} (\texttt{chernoff}). For $s > 0$, Markov's inequality applied
to $e^{sD}$ gives $\Pr(D_k \geq \tau) \leq \exp(K^{\mathrm{t}} - s\tau)$. The row
is
\begin{equation}
  K_k - s\,\tau \leq \ln\varepsilon.
  \label{eq:rf_chernoff}
\end{equation}
\emph{Validity.} $K^{\mathrm{t}} \leq K_k$, so
$\Pr(D_k > \tau) \leq \exp(K_k - s\tau) \leq \varepsilon$. The row bounds $K$ by
$s\tau - \kappa$, its natural range; $K \geq 0$, because
$K(X) \geq s\,\mathbb{E}X \geq 0$ by Jensen's inequality and A0. A known zero
damage satisfies the row only if $s\tau \geq \kappa$ (input check
$s > \ln(1/\varepsilon)/\tau$, Section~\ref{sec:consistency_checks}). A
replacement day satisfies it only if $K^{\new} \leq s\tau - \kappa$; the input
checks require every replacement state to satisfy the cell's row.
```

### M-B-51: Rainflow: Bernstein tangent condition, the reason

- From: Rainflow Model, Linear forms, lines 1116-1133 of the frozen copy.
- Kind: derivation.
- Suggested destination: Implementation details.
- Spec keeps: the tangent and pwl conditions as requirements.

```latex
Each $f$ is a parabola in $\mu$ with a positive leading coefficient, so it is
convex. For Bernstein with $b > 0$, the tangent point must satisfy
$\mu_r < \sqrt{\tau(\tau - 2c)}$, and the \texttt{pwl} form needs
$\tau/(2P) < \sqrt{\tau(\tau - 2c)}$ for its first segment, with
$P = \texttt{pwl\_points}$ (input checks of Section~\ref{sec:consistency_checks}).
The reason: a tangent is linear in $\mu$, so its largest value on
$[0, \tau - c]$ is at an end point. At $\tau - c$ it is at most
$f(\tau - c) = -c^2/(2\kappa) < 0$. At $0$ it equals
$f(0) - \mu_r^2/(2\kappa) = \bigl(\tau(\tau - 2c) - \mu_r^2\bigr)/(2\kappa)$,
because the tangent lies below $f$ by $c_2(\mu - \mu_r)^2$ with
$c_2 = 1/(2\kappa)$ (Section~\ref{sec:linear_forms}). So the tangent row
admits a point with $v \geq 0$ exactly when $\mu_r^2 \leq \tau(\tau - 2c)$, and at
equality only $\mu = v = 0$. Since $\sqrt{\tau(\tau - 2c)} < \tau - c$, this
covers the case $\mu_r \geq \tau - c$, in which the tangent has a nonnegative
slope. The \texttt{pwl} form accepts the points below the largest of its tangents
(Section~\ref{sec:linear_forms}), a convex function; at $\mu = 0$ the largest is
the tangent at the midpoint $\tau/(2P)$ of the first segment, so the same argument
applies with $\mu_r = \tau/(2P)$.
```

### M-B-52: Rainflow: facts (M1)-(M3)

- From: Rainflow Model, lines 1138-1149 of the frozen copy.
- Kind: statement.
- Suggested destination: Stochastic orderings: loop constraint for each degradation model.
- Spec keeps: nothing.

```latex
\textbf{Facts (M1)--(M3) of Section~\ref{sec:encoding}.}
(M1) The repair factors $1-\rho$ (for $\mu$ and $K$) and $(1-\rho)^2$ (for $v$ and
$R$) lie in $[0, 1]$, and the increments $\delta\mu_k$, $\delta v_k$, $\delta R_k$,
$\delta K_k$ are non-negative ($K^{\inc} \geq s\,\mu^{\inc} \geq 0$ by Jensen's
inequality and A0).
(M2) $\theta_0 \leq \xi_0$, as in Section~\ref{sec:dyn_gaussian}; on every later
day $\theta_k \leq \varphi_k(\theta_{k-1})$, with equality for $\mu$, $v$ and $R$
and an inequality for $K$ (Proposition~\ref{prop:rf_maps}).
(M3) If values that satisfy the row of the cell's bound are at least the true
statistics of day $k$, then $\Pr(D_k > \tau) \leq \varepsilon$ (the validity
paragraphs above). The loop row of Section~\ref{sec:loop} covers $\mu$, the extra
state of the bound and, under $\ARDone$, their anchors.
```
