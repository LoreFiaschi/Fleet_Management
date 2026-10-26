# Blocks moved out of the spec by writer A (26 Oct 2026)

Source: `fragments_A.tex` as it stood before the split into specification and scoping (backup in the session scratchpad, `spec_A/fragments_A_before_split.tex`). Each block is copied verbatim, as whole source lines. Labels inside the blocks (`lem:a_monotone`, `prop:a_*`, `thm:a_loop`, `def:a_laws`) no longer exist in the spec; the scoping writer can keep or rename them.

### M-A-1: Gaussian versus Wiener; rainflow versus the parametric models
- From: Project Overview & Goals (sec:overview), after the list of models.
- Kind: comparison.
- Suggested destination: Overview of the problem: Damage dynamics.
- Spec keeps: nothing (the list of the five models stays).

```latex
Both the Gaussian and the Wiener model use Gaussian increments. The Gaussian model
tracks a variance that a repair reduces. The Wiener model fits components whose
uncertainty does not depend on the route and is not reduced by a repair. When in
doubt, use the Gaussian model. The rainflow model makes no distributional
assumption: its certificate holds for every increment law with the measured
summaries, at the price of a looser bound than an exact distribution function.
```

### M-A-2: Cost terms with their reasons
- From: Objective Function (sec:objective), paragraph "Cost terms".
- Kind: rationale.
- Suggested destination: Overview of the problem: Cost function.
- Spec keeps: the cost terms as definitions (C_M per train per depot day; C_R per unit of removed mean degradation; C_rep required; C_D u added once).

```latex
\textbf{Cost terms.}
\begin{itemize}[nosep,leftmargin=1.5em]
  \item $C_M \geq 0$, a scalar, is paid once per train per depot day ($x_{i0k} = 1$),
        whatever the number of components repaired or replaced on that day. It
        prices the visit to the depot, for example the train out of service and the
        set-up. An idle day (Section~\ref{sec:assignment}) costs nothing, so the
        number of depot days is a costed decision.
  \item $C_{R,i\ell} \geq 0$ is paid per unit of mean degradation removed by a
        repair (Section~\ref{sec:repair_cost}).
  \item $C_{\mathrm{rep},i\ell} \geq 0$ is paid per replacement. The input is
        required; it has no default.
  \item $C_D \geq 0$ weights the damage penalty $u$
        (Section~\ref{sec:damage_reg}). The term is added once in every mode,
        because $u$ is the maximum over the whole horizon.
\end{itemize}
```

### M-A-3: Meaning of the objective modes, with an example and the upper-bound remark
- From: Objective Function (sec:objective), paragraph "Meaning of the modes".
- Kind: rationale.
- Suggested destination: Overview of the problem: Cost function.
- Spec keeps: nothing (Eq. eq:a_objective, one line per mode).

```latex
\textbf{Meaning of the modes.}
\begin{itemize}[nosep,leftmargin=1.5em]
  \item \texttt{total}: the cost of the $T$ modelled days.
  \item \texttt{operating\_average}: the cost per day of the operating phase. It is
        the long-run cost per day of the endless schedule
        (Section~\ref{sec:loop}), in which the cost of the transitory phase spreads
        over infinitely many days. It compares solves with different $H_2$
        (Section~\ref{sec:sweeps}). The transitory costs do not enter the
        objective in this mode: the solver chooses the transitory schedule for
        feasibility only.
  \item \texttt{evaluation\_total}: the projected cost over $E$ days: the transitory
        phase once, then $(E - H_1)/H_2$ repetitions of the operating phase, a
        fraction of a repetition counted pro rata. It also prices the transitory
        phase, so it compares solves with different $H_1$.
\end{itemize}
Example: $H_1 = 5$, $H_2 = 10$, $E = 365$. Then \texttt{evaluation\_total} weights
the operating cost by $36$, and \texttt{operating\_average} by $1/10$.

The objective value is computed from the tracked values. It is therefore an upper
bound on the cost of the schedule computed from the reference values
(Section~\ref{sec:encoding}).
```

### M-A-4: Introduction to the constraints
- From: Constraints (sec:constraints), first paragraph.
- Kind: remark.
- Suggested destination: Overview of the problem: Abstract problem formulation.
- Spec keeps: one sentence: the rows hold for every day k = 1..T unless a section says otherwise.

```latex
The constraints have three groups. The fleet rows (assignment and depot days) do not
depend on the degradation models. The common rules of the dynamics (repair laws,
encoding, replacement, linear forms of the reliability rows) hold for every model;
the five model sections apply them and add each model's tracked quantities, day
maps and reliability row. The repair cost, the damage regularisation and the loop
constraint close the problem. All rows hold for every day $k = 1, \ldots, T$ unless a
section says otherwise.
```

### M-A-5: Idle days and the pricing of depot days
- From: Assignment Constraints (sec:assignment), paragraph "Idle days".
- Kind: rationale.
- Suggested destination: Overview of the problem: Scheduling and maintenance variables.
- Spec keeps: one sentence: an idle day is an all-zero assignment row; depot and idle days add no increment.

```latex
\emph{Idle days.} A train is idle on day $k$ when its assignment row is all zeros:
$x_{ijk} = 0$ for every $j \in \{0, 1, \ldots, M\}$. An idle train flies no mission
and is not in the depot. Idle days and depot days add no increment, because every
day aggregate $\delta\mu_{i\ell k}$, $\delta v_{i\ell k}$, $\delta R_{i\ell k}$,
$\delta K_{i\ell k}$ sums over missions $j \geq 1$ only. A depot day must be chosen
explicitly ($x_{i0k} = 1$), and repairs and replacements need one
(Section~\ref{sec:depot}). Because Eq.~\eqref{eq:a_one_activity} is an inequality,
the choice between a depot day and an idle day is free, and the term $C_M \sum
x_{i0k}$ of the objective prices it.
```

### M-A-6: One depot row instead of three
- From: Depot-Day Compatibility (sec:depot), remark.
- Kind: remark.
- Suggested destination: Overview of the problem: Scheduling and maintenance variables.
- Spec keeps: nothing.

```latex
\emph{Remark (one row instead of three).} Eq.~\eqref{eq:a_depot} has the same binary
solutions as the three rows $x^m \leq x_{i0k}$, $x^r \leq x_{i0k}$, $x^m + x^r \leq 1$.
Its linear relaxation is tighter: the point $x_{i0k} = x^m = x^r = 0.5$ satisfies the
three rows but not Eq.~\eqref{eq:a_depot}. With $0 \leq x^m, x^r$ and $x_{i0k} \leq 1$,
Eq.~\eqref{eq:a_depot} describes the convex hull of the four binary solutions
$(x_{i0k}, x^m, x^r) \in \{(0,0,0), (1,0,0), (1,1,0), (1,0,1)\}$.
```

### M-A-7: Names and sources of the repair laws
- From: Repair Laws (sec:repair_laws), paragraph "Names and sources".
- Kind: literature.
- Suggested destination: Maintenance strategies: reduction-based models; connection to the fleet management formulation.
- Spec keeps: nothing (the laws stay as formulas with their input values).

```latex
\emph{Names and sources.} $\ARDone$, ``arithmetic reduction of deterioration of
order 1'', is the law of \cite[$\S$3.1, eq.~(4)]{mercier2019}. That source measures the
removed degradation from the level just after the last maintenance, as
Eq.~\eqref{eq:a_ard1} does. The name $\ARDinf$ is an analogy, not a citation.
Doyen and Gaudoin define the classes of arithmetic reduction of age (ARA), among them
ARA$_\infty$ \cite[eq.~(20), p.~11]{doyen2004} and ARA$_1$
\cite[eq.~(22), p.~12]{doyen2004}. They reduce a virtual age, not a degradation
level. ARA$_\infty$ reduces the whole virtual age; ARA$_1$ reduces only
the age accumulated since the previous repair. The subscript is the memory of the
class: the largest number of previous failure times that enter the failure
intensity \cite[p.~8]{doyen2004}. $\ARDinf$ applies the rule of ARA$_\infty$ to the
whole degradation level. No law of this specification acts on a virtual age.
```

### M-A-8: Repair maps: derivation of the factor of the mean
- From: Repair Laws (sec:repair_laws), paragraph "Repair maps of the tracked quantities".
- Kind: derivation.
- Suggested destination: Maintenance strategies: reduction-based models; connection to the fleet management formulation.
- Spec keeps: Eq. eq:a_psi and the factors 1-rho (mean, K), (1-rho)^2 (variance, R), 1 (Wiener variance).

```latex
\textbf{Repair maps of the tracked quantities.} A model tracks moments or other
summaries of $D$, not $D$ itself. For a tracked quantity $s$ with anchor $g$, a
repair on day $k$ applies the repair map
\begin{equation}
  \psi(s_{k-1}, g_{k-1}) =
  \begin{cases}
    a\, s_{k-1} & \ARDinf, \\
    a\, s_{k-1} + (1-a)\, g_{k-1} & \ARDone,
  \end{cases}
  \label{eq:a_psi}
\end{equation}
with a repair factor $a \in [0, 1]$. For the mean, $a = 1 - \rho_{i\ell}$, because
expectation is linear: Eq.~\eqref{eq:a_ard1} gives $\mathbb{E}[D^+] =
(1-\rho)\,\mathbb{E}[D^-] + \rho\,\mathbb{E}[D^{\mathrm{last}}]$. For a variance and
for $R$ the factor is $(1-\rho_{i\ell})^2$, and for $K$ it is $1-\rho_{i\ell}$; the
model sections derive each factor. The Wiener variance has the factor $a = 1$
under both laws: a repair acts on the Wiener mean only and leaves the variance
unchanged.
```

### M-A-9: Anchors, with a worked example
- From: Repair Laws (sec:repair_laws), paragraph "Anchors".
- Kind: example.
- Suggested destination: Maintenance strategies: reduction-based models; connection to the fleet management formulation.
- Spec keeps: Eq. eq:a_anchor, the starting-anchor inputs, their defaults and the range 0 <= g_0 <= s_0.

```latex
\textbf{Anchors.} Under $\ARDone$ each tracked quantity has an anchor $g$: the value
of the quantity just after the last intervention. The Wiener variance is the
exception: with $a = 1$ its map does not use an anchor, and none is built. The
anchor recursion is
\begin{equation}
  g_0 = \text{starting anchor}, \qquad
  g_k = \begin{cases}
    s_k & \text{on a repair or replacement day}, \\
    g_{k-1} & \text{otherwise},
  \end{cases}
  \label{eq:a_anchor}
\end{equation}
built by the rows Eq.~\eqref{eq:a_bigm_anchor} or Eq.~\eqref{eq:a_ind_anchor}. The
starting anchors $g^{\mu}_{i\ell 0}$, $g^{v}_{i\ell 0}$, $g^{R}_{i\ell 0}$,
$g^{K}_{i\ell 0}$ come from the optional inputs \texttt{anchor\_mu\_0},
\texttt{anchor\_v\_0}, \texttt{anchor\_R\_0}, \texttt{anchor\_K\_0}. The default of
each is the initial value of its quantity. Each must satisfy $0 \leq g_0 \leq s_0$
(Section~\ref{sec:consistency_checks}). Example: $\mu_{k-1} = 0.6$, $g_{k-1} = 0.2$,
$\rho = 0.5$. A repair gives $\mu_k = 0.6 - 0.5\,(0.6 - 0.2) = 0.4$ and $g_k = 0.4$.
On the next day the gap $\mu - g$ is $0$, and it grows again with each mission.
```

### M-A-10: Default of the starting anchor
- From: Repair Laws (sec:repair_laws), remark.
- Kind: rationale.
- Suggested destination: Maintenance strategies: reduction-based models; connection to the fleet management formulation.
- Spec keeps: nothing.

```latex
\emph{Remark (default of the starting anchor).} Damage only grows between
interventions. So the true level after the last intervention before the horizon lies
between the level of a new part and the initial level. A larger anchor gives a
weaker repair and a higher state, because $\psi$ is non-decreasing in $g$. The
default $g_0 = s_0$ therefore never overstates what a repair achieves; the input
gives the exact value when the maintenance history is known. Example: $\mu_0 =
0.5$, $\rho = 0.5$, and a repair on day~1 before any mission. With the starting
anchor $0$ the mean drops to $0.25$; with the default $\mu_0$ it stays at $0.5$, and
the repair removes only damage accumulated within the horizon.
```

### M-A-11: Laws per model, with reasons
- From: Repair Laws (sec:repair_laws), paragraph "Laws per model" and its table.
- Kind: rationale.
- Suggested destination: Maintenance strategies: reduction-based models; connection to the fleet management formulation; Stochastic orderings: within each degradation family.
- Spec keeps: one sentence: every model accepts both laws.

```latex
\textbf{Laws per model.} Every model accepts both laws. The table below gives the
reason for each model.

\begingroup\small
\begin{longtable}{@{}lllp{7.4cm}@{}}
\toprule
Model & $\ARDinf$ & $\ARDone$ & Reason or note \\
\midrule
Gaussian & yes & yes & both laws keep the state Gaussian, with exact moments \\
Inverse Gaussian & yes & yes & the cap holds for any repair weights and any
  dependence between the terms (Section~\ref{sec:dyn_ig}) \\
Wiener & yes & yes & on the mean only; the variance changes only by replacement \\
Gamma & yes & yes & Section~\ref{sec:dyn_gamma} \\
Rainflow, every bound & yes & yes & Chernoff under $\ARDone$ uses a conservative
  recursion for $K$ (Section~\ref{sec:dyn_rainflow}) \\
\bottomrule
\end{longtable}
\endgroup
```

### M-A-12: Encoding: introduction and one-sided rows, with an example
- From: Encoding of the Dynamics (sec:encoding), first two paragraphs.
- Kind: rationale.
- Suggested destination: Overview of the problem: Damage dynamics.
- Spec keeps: one sentence: every dynamics row is one-sided; replacement and anchor rows are equalities.

```latex
Each day, every tracked quantity of a cell follows one rule: a mission adds an
increment, a repair applies the repair map of Section~\ref{sec:repair_laws}, and a
replacement resets the quantity (Section~\ref{sec:replacement}). This subsection
states these rules once, for every model and both repair laws. The model sections
give, for each tracked quantity, its increment, its repair factor and its natural
range.

\textbf{One-sided rows.} Every dynamics row is an inequality ($\geq$), not an
equality. The solver may choose a value above the result of the rule, never a
value below it. So a tracked quantity is an upper bound on the value that the rule
gives. Example: with $\mu_{i\ell,k-1} = 0.3$ and a mission increment of $0.2$, the
rule gives $0.5$; the rows accept $\mu_{i\ell k} = 0.5$ and also $\mu_{i\ell k} =
0.6$. Replacement rows and anchor updates are equalities.
```

### M-A-13: Validity of the big-M constants
- From: Encoding of the Dynamics (sec:encoding), remark.
- Kind: proof.
- Suggested destination: Implementation details.
- Spec keeps: nothing (the natural ranges stay; the range checks are in sec:consistency_checks).

```latex
\emph{Remark (validity of the big-M constants).} Every tracked quantity and every
anchor lies in $[0, M_s]$. The initial values, the starting anchors and the
replacement values lie in $[0, M_s]$ too, by the input checks of
Section~\ref{sec:consistency_checks}. An intervention day is a depot day
(Section~\ref{sec:depot}), so $\delta s_k = 0$ on it. A relaxed row then cuts no
point: Eq.~\eqref{eq:a_bigm_carry} with $e_k = 1$ reads
$s_k \geq s_{k-1} - M_s$, whose right-hand side is at most $0$;
Eq.~\eqref{eq:a_bigm_repair} with $x^m_k = 0$ has a right-hand side
$\psi - M_s \leq 0$, because $\psi$ is a convex combination of values in
$[0, M_s]$; Eq.~\eqref{eq:a_bigm_repl} with $x^r_k = 0$ allows
$s_k \in [s^{\new} - M_s,\, s^{\new} + M_s] \supseteq [0, M_s]$; the relaxed anchor
rows allow every difference in $[-M_s, M_s]$.
```

### M-A-14: Choice of the default encoding
- From: Encoding of the Dynamics (sec:encoding), remark.
- Kind: rationale.
- Suggested destination: Implementation details.
- Spec keeps: nothing.

```latex
\emph{Remark (choice of the default form).} Gurobi accepts a binary as integral
within its integrality tolerance \texttt{IntFeasTol} (default $10^{-5}$). A big-M
row can therefore leak by at most $10^{-5} M_s$. Because $M_s$ is the quantity's own
range, this is a relative error of $10^{-5}$. Indicator constraints have no such
leak, and Gurobi usually needs more time to solve them. Which form solves this
model faster has not been measured.
```

### M-A-15: Safety of the one-sided rows: objects of the argument
- From: Encoding of the Dynamics (sec:encoding), subsubsection "Safety of the one-sided rows", first paragraph.
- Kind: statement.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: nothing.

```latex
\subsubsection{Safety of the one-sided rows}

Fix a schedule, that is, values of every binary, and one cell. The following
objects are used here and in Section~\ref{sec:loop}.
\begin{itemize}[nosep,leftmargin=1.5em]
  \item The \emph{full tracked state} $\xi_k$ collects the values on day $k$ of every
        tracked quantity of the cell and of every anchor. $\xi_0$ holds the initial
        values and the starting anchors.
  \item The \emph{day map} $\varphi_k$ gives, for each coordinate, the value that the
        rule of day $k$ assigns: $s + \delta s_k$ and $g$ on a day without
        intervention; $\psi(s, g)$ for both $s$ and $g$ on a repair day; $s^{\new}$
        for both on a replacement day.
  \item The \emph{reference trajectory} is $\zeta_0 = \xi_0$,
        $\zeta_k = \varphi_k(\zeta_{k-1})$: the rows with every inequality replaced
        by an equality.
  \item The \emph{true statistics} $\theta_k$ are the true values on day $k$ of the
        tracked quantities (the mean, the variance, $R$ or the CGF of the damage
        $D_{i\ell k}$) and of their anchors (their values just after the last
        intervention).
  \item The \emph{certificate set} $\mathcal{C}_{i\ell}$ is the set of values of the
        tracked quantities at which the reliability row of the cell holds in its
        exact form, side rows included. Anchors do not enter it.
\end{itemize}
```

### M-A-16: Facts (M1)-(M3) of every model and the certificate rows in chance-bound form
- From: Encoding of the Dynamics (sec:encoding), safety subsubsection.
- Kind: statement.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: nothing.

```latex
Every model section states three facts, labelled (M1)--(M3) there:
\begin{enumerate}[nosep,leftmargin=2em,label=(M\arabic*)]
  \item Each repair factor $a$ lies in $[0,1]$ and each increment
        $\delta s_k$ is non-negative.
  \item On day $0$ the true statistics are at most the initial values and starting
        anchors, $\theta_0 \leq \xi_0$. On every later day the true statistic is at
        most the day map applied to the true statistics of the day before:
        $\theta_k \leq \varphi_k(\theta_{k-1})$. Every map holds with equality
        except the map of the Chernoff quantity $K$, which is an upper bound
        (Section~\ref{sec:dyn_rainflow}).
  \item If values $s \in \mathcal{C}_{i\ell}$ are at least the true statistics of
        day $k$, then $\Pr(D_{i\ell k} > \tau_{i\ell}) \leq \varepsilon_{i\ell}$.
\end{enumerate}
The set $\mathcal{C}_{i\ell}$ is closed, and it is a down-set: if $s \in
\mathcal{C}_{i\ell}$ and $0 \leq s' \leq s$ componentwise, then $s' \in
\mathcal{C}_{i\ell}$. The reason is that every reliability row, written in
chance-bound form, reads ``a continuous function of the tracked quantities,
non-decreasing in each of them, is at most a constant'':

\begingroup\small
\begin{longtable}{@{}lll@{}}
\toprule
Model or bound & Row in chance-bound form & Section \\
\midrule
Gaussian, Wiener & $\mu + \kappa_G\sqrt{v} \leq \tau$ &
  \ref{sec:dyn_gaussian}, \ref{sec:dyn_wiener} \\
Inverse Gaussian; Gamma & $\mu \leq \bar\mu$; $\mu \leq A_1(\beta^*\tau,\varepsilon)/\beta^*$ &
  \ref{sec:dyn_ig}, \ref{sec:dyn_gamma} \\
Rainflow, Markov & $\mu \leq \varepsilon\,\tau$ & \ref{sec:dyn_rainflow} \\
Rainflow, Cantelli & $\mu + \kappa_C\sqrt{v} \leq \tau$ & \ref{sec:dyn_rainflow} \\
Rainflow, Hoeffding & $\mu + \sqrt{\kappa R/2} \leq \tau$ & \ref{sec:dyn_rainflow} \\
Rainflow, Bernstein & $\mu + c + \sqrt{c^2 + 2\kappa v} \leq \tau$ & \ref{sec:dyn_rainflow} \\
Rainflow, Chernoff & $K \leq s_{i\ell}\,\tau + \ln\varepsilon$ & \ref{sec:dyn_rainflow} \\
\bottomrule
\end{longtable}
\endgroup
```

### M-A-17: Lemma lem:a_monotone and Proposition prop:a_domination, with proofs
- From: Encoding of the Dynamics (sec:encoding), safety subsubsection.
- Kind: proof.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: one sentence: each tracked value is an upper bound on its reference value (pointer).

```latex
\begin{lemma}[Monotone day maps]
\label{lem:a_monotone}
Each day map $\varphi_k$ is non-decreasing in each coordinate of the full tracked
state, anchors included.
\end{lemma}
\begin{proof}
Let $\xi \leq \xi'$ componentwise. Each coordinate of $\varphi_k$ has one of the
forms $s + \delta s_k$, $g$, $s$, $a\,s$, $a\,s + (1-a)\,g$ or $s^{\new}$. By (M1),
$a \geq 0$ and $1 - a \geq 0$. Therefore each coordinate of $\varphi_k(\xi)$ is at
most the same coordinate of $\varphi_k(\xi')$.
\end{proof}

\begin{proposition}[Domination]
\label{prop:a_domination}
Every feasible solution satisfies $\xi_k \geq \varphi_k(\xi_{k-1})$ and
$\theta_k \leq \zeta_k \leq \xi_k$ for $k = 1, \ldots, T$.
\end{proposition}
\begin{proof}
At binary points the rows of day $k$ read: $s_k \geq \varphi_k(\xi_{k-1})_s$ for
every tracked quantity $s$, with equality on a replacement day; $g_k = s_k \geq
\varphi_k(\xi_{k-1})_g$ on an intervention day; $g_k = g_{k-1} =
\varphi_k(\xi_{k-1})_g$ on any other day. So $\xi_k \geq \varphi_k(\xi_{k-1})$.
Let $\theta_{k-1} \leq \zeta_{k-1} \leq \xi_{k-1}$ (true for $k = 1$ by (M2), since
$\zeta_0 = \xi_0$). Then, by (M2) and Lemma~\ref{lem:a_monotone},
\[
  \theta_k \leq \varphi_k(\theta_{k-1}) \leq \varphi_k(\zeta_{k-1}) = \zeta_k
  \leq \varphi_k(\xi_{k-1}) \leq \xi_k .
\]
Therefore $\theta_k \leq \zeta_k \leq \xi_k$ for every $k$ by induction.
\end{proof}
```

### M-A-18: Consequence on the horizon
- From: Encoding of the Dynamics (sec:encoding), safety subsubsection.
- Kind: statement.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: nothing.

```latex
\emph{Consequence on the horizon.} On every day $k = 1, \ldots, T$ the tracked
values satisfy the reliability row in the form chosen for the cell. Every form lies
inside $\mathcal{C}_{i\ell}$ (Section~\ref{sec:linear_forms}), so $\xi_k \in
\mathcal{C}_{i\ell}$. By Proposition~\ref{prop:a_domination}, $\theta_k \leq \xi_k$,
so (M3) gives $\Pr(D_{i\ell k} > \tau_{i\ell}) \leq \varepsilon_{i\ell}$.
Theorem~\ref{thm:a_loop} (Section~\ref{sec:loop}) extends this to every day of
every later repetition of the operating phase.
```

### M-A-19: What the one-sided rows accept; Proposition prop:a_accept with proof
- From: Encoding of the Dynamics (sec:encoding), safety subsubsection.
- Kind: proof.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: nothing.

```latex
\textbf{What the one-sided rows accept.} Call a schedule \emph{safe} for a cell when
the reference trajectory of the endless schedule (the transitory phase once, then
the operating phase repeated without end, Section~\ref{sec:loop}) lies in
$\mathcal{C}_{i\ell}$ and in the natural ranges on every day. The one-sided rows
relate to this property as follows.
\begin{itemize}[nosep,leftmargin=1.5em]
  \item \emph{They never accept an unsafe schedule} (Theorem~\ref{thm:a_loop}).
  \item \emph{They accept every schedule that equality rows accept}, and more.
        Equality rows force $\xi_k = \zeta_k$; Proposition~\ref{prop:a_accept}~(a)
        shows that such a schedule satisfies the one-sided rows too.
        The first example below shows a safe schedule that only the one-sided
        rows accept.
  \item \emph{They may reject a safe schedule}, as equality rows may.
        Proposition~\ref{prop:a_accept} (both parts) excludes this for a cell in the exact
        form with one tracked quantity, the law $\ARDinf$ and no replacement on day
        $H_1$. Otherwise a rejection can occur in three cases: when tracked
        quantities move in opposite directions over the first repetition (the
        second example below); when the cell is replaced on day $H_1$, because the
        first repetition then starts from the fixed replacement state; or when a
        linear form is stricter than the exact row.
\end{itemize}

\begin{proposition}[Acceptance]
\label{prop:a_accept}
Fix a schedule and a cell, use the exact reliability form, and let the schedule be
safe for the cell. Let $\bar\zeta = \zeta_{H_1}$ and let $\Pi$ be the composition
of the day maps of the operating phase (Section~\ref{sec:loop}).
\begin{enumerate}[nosep,label=(\alph*)]
  \item If $\Pi(\bar\zeta) \leq \bar\zeta$, the rows of the cell, the loop row
        included, accept the schedule with $\xi_k = \zeta_k$ for every $k$.
  \item If $\Pi(\bar\zeta) \geq \bar\zeta$, the cell has no anchor (law
        $\ARDinf$), and the cell is not replaced on day $H_1$, the rows of the cell
        accept the schedule.
\end{enumerate}
\end{proposition}
\begin{proof}
(a) The values $\xi_k = \zeta_k$ satisfy every row of day $k$ with equality at
binary points. They lie in $\mathcal{C}_{i\ell}$ because the schedule is safe. The
loop row reads $\zeta_T = \Pi(\bar\zeta) \leq \bar\zeta = \zeta_{H_1}$.

(b) Let $y_n = \Pi^n(\bar\zeta)$, the reference state at the end of the $n$-th
repetition. By Lemma~\ref{lem:a_monotone} and $\Pi(\bar\zeta) \geq \bar\zeta$,
$y_{n+1} = \Pi(y_n) \geq \Pi(y_{n-1}) = y_n$. The sequence is non-decreasing and
lies in the natural ranges, so it converges to some $y^*$ with $y^* \geq
\bar\zeta$. Each day map is affine or constant, so $\Pi$ and its prefixes
$\Pi_h$ are continuous: $\Pi(y^*) = \lim_n y_{n+1} = y^*$ and
$\Pi_h(y^*) = \lim_n \Pi_h(y_n)$. Each $\Pi_h(y_n)$ is a reference value of a
later day, so it lies in the closed set $\mathcal{C}_{i\ell}$; hence
$\Pi_h(y^*) \in \mathcal{C}_{i\ell}$ for $h = 1, \ldots, H_2$, and
$y^* = \Pi_{H_2}(y^*) \in \mathcal{C}_{i\ell}$. Take $\xi_k =
\zeta_k$ for $k < H_1$, $\xi_{H_1} = y^*$ and $\xi_{H_1+h} = \Pi_h(y^*)$. Day $H_1$
is not a replacement day and the cell has no anchor, so $\xi_{H_1} = y^* \geq
\varphi_{H_1}(\xi_{H_1-1})$ satisfies the rows of day $H_1$. The later days satisfy
their rows with equality, and the loop row reads $\xi_T = \Pi(y^*) = y^* =
\xi_{H_1}$.
\end{proof}

A scalar state is always comparable with its image. So a cell in the exact form with
one tracked quantity under $\ARDinf$, not replaced on day $H_1$, accepts every safe
schedule.
Under $\ARDone$ the anchor is a second tracked quantity, and its value on day $H_1$
is fixed by the anchor rows; part (b) does not apply, and acceptance of every safe
schedule is not proven.
```

### M-A-20: Examples: one quantity accepted only by the one-sided rows; two quantities rejected
- From: Encoding of the Dynamics (sec:encoding), safety subsubsection.
- Kind: example.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: nothing.

```latex
\emph{Example (one quantity, accepted only by the one-sided rows).}
One tracked mean, law $\ARDinf$, $\rho = 0.5$. The operating phase is a mission
($\delta\mu = 0.2$) followed by a repair, so $\Pi(y) = 0.5\,(y + 0.2) = 0.5\,y +
0.1$, and $\Pi(y) \leq y$ if and only if $y \geq 0.2$. Let $\bar\zeta = 0.1$. Then
$\Pi(0.1) = 0.15 > 0.1$, so equality rows reject the schedule. The one-sided rows
accept it with $\xi_{H_1} = 0.2$: the tracked cycle is $0.2, 0.4, 0.2$, and the
reliability row is checked at $0.4$. The reference trajectory is $0.1, 0.3, 0.15,
0.35, 0.175, \ldots$; its start-of-cycle values rise to $0.2$ and its peaks rise to
$0.4$. So the schedule is safe whenever $0.4 \in \mathcal{C}_{i\ell}$. The price
is in the reported cost: each repair is charged $\rho \cdot 0.4 = 0.2$, while the
reference removal rises from $0.15$ toward $0.2$.

\emph{Example (two quantities that move in opposite directions, rejected).}
Cantelli bound, $\varepsilon = 0.1$ ($\kappa_C^2 = 9$), $\tau = 1$, $\rho = 0.5$,
law $\ARDinf$. The operating phase is a mission, which adds $0.2$ to the mean and
$0.009$ to the variance, followed by a repair. So $\Pi(\mu, v) = (0.5\,\mu + 0.1,\;
0.25\,v + 0.00225)$, with fixed point $(0.2, 0.003)$. Let $\bar\zeta = (0.5, 0)$.
\begin{itemize}[nosep,leftmargin=1.5em]
  \item Reference: over the repetitions the mean falls to $0.2$ and the variance
        rises to $0.003$. The first peak is $(0.7, 0.009)$, with $9 \cdot 0.009 =
        0.081 \leq 0.3^2 = 0.09$. Every later peak has a mean of at most $0.55$ and a
        variance below $0.012$, with $9 \cdot 0.012 = 0.108 \leq 0.45^2 = 0.2025$.
        The schedule is safe.
  \item One-sided rows: the loop row needs $\xi_{H_1} = y \geq (0.5, 0)$ with
        $\Pi(y) \leq y$, so $y_v \geq 0.003$. The tracked peak is then at least
        $(0.7, 0.012)$, where $9 \cdot 0.012 = 0.108 > 0.09$. Rejected.
  \item Equality rows: $\Pi(\bar\zeta) = (0.35, 0.00225)$ has a larger variance
        than $\bar\zeta$. Rejected as well.
\end{itemize}
```

### M-A-21: Meaning of the reported values; slack in the reported values
- From: Encoding of the Dynamics (sec:encoding), end of the safety subsubsection.
- Kind: remark.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model); Overview of the problem: Cost function.
- Spec keeps: one sentence in sec:variables: the output reports the tracked values, upper bounds on the reference values (pointer).

```latex
\textbf{Meaning of the reported values.} The output reports the tracked values
$\xi_k$, not the reference values. By Proposition~\ref{prop:a_domination} each
reported state and anchor is an upper bound on its reference value, and so on its
true statistic. The costs computed from them are upper bounds on the costs computed
from the reference values: the repair cost (Proposition~\ref{prop:a_repair_cost}) and
the damage penalty (Section~\ref{sec:damage_reg}). Against the true costs,
Section~\ref{sec:repair_cost} states one exception, for a starting anchor above the
true one.

\emph{Remark (slack in the reported values).} The minimisation removes slack only
where slack costs something: a larger mean raises the damage penalty $C_D\,u$ where
$u$ is binding, and raises the repair cost $C_R\,z$ of every later repair. A larger
variance, $R$ or $K$ tightens the reliability row and so restricts the schedule.
Elsewhere the solver may return any value above the reference value. With $C_D = 0$
the damage penalty is absent (Section~\ref{sec:damage_reg}), and the reported states
may carry more slack. Slack never affects safety (Theorem~\ref{thm:a_loop}).
```

### M-A-22: Replacement value semantics and the reason for the replacement-state check
- From: Replacement (sec:replacement), after the list of replacement values.
- Kind: rationale.
- Suggested destination: Overview of the problem: Damage dynamics.
- Spec keeps: the requirement: each replacement value in its natural range; the replacement state satisfies every row of a replacement day in the form built at the cell.

```latex
A replacement value describes the damage of the new part: a known damage when its
variance is $0$, a random one otherwise. The rows are the equalities
Eq.~\eqref{eq:a_bigm_repl} or the indicator rows Eq.~\eqref{eq:a_ind_state}. The
input checks of Section~\ref{sec:consistency_checks} require each replacement value
to lie in the natural range of its quantity. They also require the replacement state
to satisfy every row that holds on a replacement day, in the form built at the cell:
the reliability row in the form of \texttt{reliability\_impl} (for \texttt{pwl}, the
cap of the segment that contains $\mu^{\new}$), its side row, and $v^{\new} \leq
V^{\mathrm{user}}$ where \texttt{v\_max\_user} is given. A linear form is stricter
than the exact row (Proposition~\ref{prop:a_tangent}). Without the check, a
replacement state that passes only the exact row would make every replacement of the
cell infeasible.
```

### M-A-23: Replacement cost and the replacements that the loop row requires
- From: Replacement (sec:replacement), paragraph "Cost".
- Kind: remark.
- Suggested destination: Overview of the problem: Cost function; Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: the same content in two sentences, with a pointer instead of the proposition reference.

```latex
\textbf{Cost.} A replacement costs $C_{\mathrm{rep},i\ell}$ and nothing else: the
repair cost $z_{i\ell k}$ is $0$ on a replacement day
(Section~\ref{sec:repair_cost}). A cell may be replaced on any depot day of its
train, any number of times. The loop row can require at least one replacement in
the operating phase: for every $\ARDone$ cell with $\rho < 1$ that accumulates damage
(Proposition~\ref{prop:a_ard1_replacement}), and for the Wiener variance
(Section~\ref{sec:dyn_wiener}).
```

### M-A-24: Quadratic rows: constants, side row and convexity
- From: Linear Forms of the Reliability Rows (sec:linear_forms), after Eq. eq:a_quad.
- Kind: rationale.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution (and two refinements).
- Spec keeps: the table of constants and the formula of f'.

```latex
with the constants of the table below; the model sections derive each row. The
third relation is the side row; it
selects the branch of the parabola on which the row is a valid certificate. Since
$c_2 > 0$, $f$ is convex in $\mu$, with $f'(\mu) = -2c_2(\tau - \mu) - c_1$.
```

### M-A-25: Exact form: why the row is nonconvex
- From: Linear Forms (sec:linear_forms), bullet exact.
- Kind: rationale.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution (and two refinements).
- Spec keeps: the row is a nonconvex quadratic constraint; NonConvex = 2 when a cell has a quadratic row in this form.

```latex
  \item \texttt{exact}: Eq.~\eqref{eq:a_quad} as written. The feasible set
        $\{(\mu, \chi) : \chi \leq f(\mu)\}$ lies below a convex function, so it is not
        convex: the row is a nonconvex quadratic constraint. Gurobi's parameter
        \texttt{NonConvex} is set to $2$ whenever at least one cell has a quadratic
        row in this form.
```

### M-A-26: Tangent form: example at tangent_ref = 0, exclusion of 1, derivation of the Bernstein condition
- From: Linear Forms (sec:linear_forms), bullet tangent.
- Kind: derivation.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution (and two refinements).
- Spec keeps: the range [0, 1) of tangent_ref and the Bernstein requirement mu_r < sqrt(tau(tau - 2c)).

```latex
        At \texttt{tangent\_ref} $= 0$ the Gaussian row becomes
        $\kappa_G^2\, v + 2\tau\mu \leq \tau^2$.
        The value $1$ is excluded: for $c_1 = 0$ the tangent at $\tau$ is
        $\chi \leq 0$. At a Bernstein cell the tangent point must also satisfy
        $\mu_r < \sqrt{\tau\,(\tau - 2c_{i\ell})}$
        (Section~\ref{sec:consistency_checks}). By
        Proposition~\ref{prop:a_tangent}, the tangent row at $\mu = 0$ reads
        $v \leq f(0) - c_2\,\mu_r^2 = \bigl(\tau(\tau - 2c) - \mu_r^2\bigr)/(2\kappa)$,
        and for $\mu_r < \tau - c$ the row decreases in $\mu$. So a point with
        $v \geq 0$ and $\mu \geq 0$ exists exactly when $\mu_r^2 \leq
        \tau(\tau - 2c)$, and only $\mu = v = 0$ at equality. Since
        $\sqrt{\tau(\tau - 2c)} < \tau - c$, the condition also excludes the case
        $\mu_r \geq \tau - c$, where the tangent has a non-negative slope and lies
        below $f(\tau - c) < 0$ on $[0, \tau - c]$ (Section~\ref{sec:dyn_rainflow}).
```

### M-A-27: pwl form: derivation of the Bernstein first-segment condition
- From: Linear Forms (sec:linear_forms), bullet pwl.
- Kind: derivation.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution (and two refinements).
- Spec keeps: the requirement tau/(2P) < sqrt(tau(tau - 2c)) at Bernstein cells.

```latex
        with no binaries. At a Bernstein cell the tangent point of the first
        segment must satisfy the same condition, $\tau/(2P) < \sqrt{\tau\,(\tau -
        2c_{i\ell})}$ (Section~\ref{sec:consistency_checks}). Otherwise no segment
        admits a point with $v > 0$: the first segment fails as above, and on every
        later segment $\mu \geq \tau/P > \tau - 2c$, where $f(\mu) < 0$.
```

### M-A-28: Proposition prop:a_tangent (safety of the linear forms) with proof
- From: Linear Forms (sec:linear_forms).
- Kind: proof.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution (and two refinements).
- Spec keeps: nothing.

```latex
\begin{proposition}[Safety of the linear forms]
\label{prop:a_tangent}
For every $\mu_r$ and every $\mu$,
$f(\mu_r) + f'(\mu_r)\,(\mu - \mu_r) = f(\mu) - c_2\,(\mu - \mu_r)^2 \leq f(\mu)$.
Therefore every point that satisfies the \texttt{tangent} or the \texttt{pwl} form
satisfies the \texttt{exact} form, and both linear forms lie inside the certificate
set $\mathcal{C}_{i\ell}$.
\end{proposition}
\begin{proof}
$f$ is a polynomial of degree $2$ in $\mu$ with leading coefficient $c_2$, so its
Taylor expansion at $\mu_r$ is exact: $f(\mu) = f(\mu_r) + f'(\mu_r)(\mu - \mu_r) +
c_2(\mu - \mu_r)^2$. Since $c_2 > 0$, the tangent lies below $f$. Each linear form
caps $\chi$ by one such tangent and keeps the side row. Therefore it implies
Eq.~\eqref{eq:a_quad}.
\end{proof}
```

### M-A-29: Accuracy of the tangent and pwl forms
- From: Linear Forms (sec:linear_forms), remark.
- Kind: remark.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution (and two refinements).
- Spec keeps: nothing.

```latex
\emph{Remark (accuracy).} The \texttt{tangent} form loses $c_2(\mu - \mu_r)^2$ in
$\chi$: it is exact at $\mu = \mu_r$ and loosest far from it. A tangent point near the
operating mean of the cell is the best choice. In the \texttt{pwl} form the loss is
$c_2(\mu - \hat\mu_p)^2 \leq c_2\,\tau^2/(4P^2)$. The midpoint of the segment that
contains $\mu$ is the nearest tangent point to $\mu$. So the \texttt{pwl} form
accepts exactly the points with $\chi \leq \max_p\, [f(\hat\mu_p) + f'(\hat\mu_p)(\mu -
\hat\mu_p)]$, the largest set that the $P$ tangents allow. The price is $P$ binaries
per cell and day.
```

### M-A-30: Cantelli example of the three forms
- From: Linear Forms (sec:linear_forms), example.
- Kind: example.
- Suggested destination: Literature review: conservative surrogates for the empirical damage-increment distribution (and two refinements).
- Spec keeps: nothing.

```latex
\emph{Example (Cantelli, $\varepsilon = 0.1$, $\kappa_C^2 = 9$, $\tau = 1$, $v =
0.004$).} The largest admissible mean is $1 - \sqrt{9 \cdot 0.004} \approx 0.810$
with \texttt{exact}; $0.482$ with \texttt{tangent} at $0$ ($9v + 2\mu \leq 1$);
$0.714$ with \texttt{tangent} at $0.5\,\tau$ ($9v \leq 0.75 - \mu$); and $0.810$
with \texttt{pwl} and $P = 8$. The tangent at $0$ never admits a mean above
$\tau/2$, even with zero variance.
```

### M-A-31: The repair-cost row on repair days and on other days; replacement days
- From: Repair Cost Variable (sec:repair_cost), after Eq. eq:a_z.
- Kind: derivation.
- Suggested destination: Overview of the problem: Cost function.
- Spec keeps: one sentence: z is 0 on a replacement day; a replacement costs C_rep only.

```latex
with $g^{\mu} \equiv 0$ under $\ARDinf$. On a repair day the first row reads $z_{i\ell
k} \geq \rho\,(\mu_{i\ell,k-1} - g^{\mu}_{i\ell,k-1})$, the removed amount of
Eq.~\eqref{eq:a_psi}. On any other day its right-hand side is at most $0$, because
$0 \leq \mu_{i\ell,k-1} - g^{\mu}_{i\ell,k-1} \leq \mu_{i\ell,k-1} \leq \tau$.

\emph{Replacement days.} A replacement day has $x^m_{i\ell k} = 0$, so $z_{i\ell k} =
0$ at the optimum. A replacement costs $C_{\mathrm{rep},i\ell}$ only. The replaced
part is discarded, so its damage does not change the price of the new part.
```

### M-A-32: No upper McCormick rows
- From: Repair Cost Variable (sec:repair_cost), remark.
- Kind: rationale.
- Suggested destination: Overview of the problem: Cost function.
- Spec keeps: one sentence: only the lower rows of Eq. eq:a_z are built.

```latex
\emph{Remark (no upper rows).} The variable $z_{i\ell k}$ appears in no row other
than Eq.~\eqref{eq:a_z}, and in the objective with a positive coefficient. The
minimisation therefore drives it to the largest of its lower bounds:
$z_{i\ell k} = \rho\,(\mu_{i\ell,k-1} - g^{\mu}_{i\ell,k-1})$ on a repair day and
$z_{i\ell k} = 0$ on any other day. The upper rows of a McCormick envelope,
$z \leq \rho\,(\mu - g^{\mu})$ and $z \leq \rho\,\tau\,x^m$, would only remove points
that are never optimal, so they are not built. A solution that is not optimal (for
example, the best one found at the time limit) may carry a larger $z$; its reported
cost is then larger, never smaller.
```

### M-A-33: Proposition prop:a_repair_cost (what z measures) with proof and the starting-anchor exception
- From: Repair Cost Variable (sec:repair_cost).
- Kind: proof.
- Suggested destination: Overview of the problem: Cost function.
- Spec keeps: one sentence: z is an upper bound on the expected removed degradation, except at the first repair when the starting anchor is above the true one (pointer).

```latex
\begin{proposition}[What $z$ measures]
\label{prop:a_repair_cost}
In every feasible solution, on every repair day $k$ of a cell, under both laws,
\[
  z_{i\ell k} \;\geq\; \rho\,\bigl(\mu_{k-1} - g^{\mu}_{k-1}\bigr)
  \;\geq\; \rho\,\bigl(\zeta^{\mu}_{k-1} - \zeta^{g}_{k-1}\bigr),
\]
where $\zeta^{\mu}$ and $\zeta^{g}$ are the mean and its anchor on the reference
trajectory (Section~\ref{sec:encoding}). When the starting anchor equals the true
mean level after the last intervention before the horizon, the right-hand side is
the expected degradation that the repair removes.
\end{proposition}
\begin{proof}
The first inequality is Eq.~\eqref{eq:a_z} with $x^m_{i\ell k} = 1$. Under $\ARDinf$,
$g^{\mu} = \zeta^{g} = 0$, and $\mu_{k-1} \geq \zeta^{\mu}_{k-1}$ by
Proposition~\ref{prop:a_domination}. Under $\ARDone$, let $t < k$ be the last
intervention day before $k$, or $t = 0$ if there is none. Then $g^{\mu}_{k-1} =
g^{\mu}_t$ and $\zeta^{g}_{k-1} = \zeta^{g}_t$.
\begin{itemize}[nosep,leftmargin=1.5em]
  \item If $t = 0$: $g^{\mu}_0 = \zeta^{g}_0$ (the starting anchor), so
        $g^{\mu}_{k-1} - \zeta^{g}_{k-1} = 0 \leq \mu_{k-1} - \zeta^{\mu}_{k-1}$.
  \item If $t \geq 1$: $g^{\mu}_t = \mu_t$ and $\zeta^{g}_t = \zeta^{\mu}_t$
        (Eq.~\eqref{eq:a_anchor}). On the days $j = t+1, \ldots, k-1$ there is no
        intervention: $\mu_j \geq \mu_{j-1} + \delta\mu_j$ and $\zeta^{\mu}_j =
        \zeta^{\mu}_{j-1} + \delta\mu_j$. So $\mu_j - \zeta^{\mu}_j$ does not
        decrease, and $\mu_{k-1} - \zeta^{\mu}_{k-1} \geq \mu_t - \zeta^{\mu}_t =
        g^{\mu}_{k-1} - \zeta^{g}_{k-1}$.
\end{itemize}
Rearranging gives $\mu_{k-1} - g^{\mu}_{k-1} \geq \zeta^{\mu}_{k-1} -
\zeta^{g}_{k-1}$. The mean recursion is linear (Section~\ref{sec:repair_laws}); with
an exact starting anchor, $\zeta^{\mu}$ and $\zeta^{g}$ are the true means, so
$\rho\,(\zeta^{\mu}_{k-1} - \zeta^{g}_{k-1}) = \mathbb{E}[D^- - D^+]$.
\end{proof}

A starting anchor above the true one, which the default may be
(Section~\ref{sec:repair_laws}), models a weaker repair. For the first repair before
any intervention of the horizon, the charged removal then follows the starting
anchor and may be below the true removal. From the first intervention on, the
reference gap $\zeta^{\mu} - \zeta^{g}$ equals the true gap, the sum of the mean
increments since that intervention, so the bound holds again.

Under both laws, then, the reported repair cost $\sum C_R\, z$ is an upper bound on
the expected repair cost of the schedule, with the exception above for a starting
anchor above the true one. Under $\ARDone$ the tracked mean and the
tracked anchor are both upper bounds, and their difference is still an upper bound
on the reference difference: the anchor carries the slack that the mean had at the
last intervention, and the slack of the mean does not decrease until the next one.
The bound is exact when the tracked mean has no slack.
```

### M-A-34: Damage penalty: the sign of C_D
- From: Damage Regularisation (sec:damage_reg), first paragraph (last sentence).
- Kind: rationale.
- Suggested destination: Overview of the problem: Cost function.
- Spec keeps: the check C_D >= 0 is in sec:consistency_checks.

```latex
built, and the input \texttt{penalty\_type} is ignored. The input check requires
$C_D \geq 0$ (Section~\ref{sec:consistency_checks}): a negative $C_D$ would reward
damage.
```

### M-A-35: Damage penalty: reason for the exact-form rule of the quadratic penalty
- From: Damage Regularisation (sec:damage_reg), bullet quadratic (last sentences).
- Kind: rationale.
- Suggested destination: Overview of the problem: Cost function.
- Spec keeps: the rule itself.

```latex
        the penalty is not built and the rule does not apply. The reason: the
        linear forms are chosen for a model without quadratic rows, and this row
        is quadratic.
```

### M-A-36: Role of the damage penalty
- From: Damage Regularisation (sec:damage_reg), paragraph "Role".
- Kind: rationale.
- Suggested destination: Overview of the problem: Cost function.
- Spec keeps: nothing.

```latex
\emph{Role.} The term prices the worst train-level damage, so the minimisation
spreads damage across the fleet. It also removes slack from the tracked means where
$u$ is binding (Section~\ref{sec:encoding}). Since $\mu_{i\ell k} \geq
\zeta^{\mu}_{i\ell k} \geq 0$ (Proposition~\ref{prop:a_domination}), both forms of $u$
are upper bounds on the same quantity computed from the reference values.
```

### M-A-37: Endless schedule and the maps of the operating phase
- From: Loop Constraint (sec:loop), paragraph "Endless schedule".
- Kind: statement.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: one sentence: the row makes every repetition of the operating phase safe (pointer).

```latex
\textbf{Endless schedule.} Repeat the operating phase without end: day $T + n H_2 +
h$, for $n \geq 0$ and $1 \leq h \leq H_2$, carries the binaries of day $H_1 + h$.
It also carries the increments of day $H_1 + h$, because the operating profile is
read cyclically with period $H_2$ (Section~\ref{sec:input_increments}). So every
repetition applies the same day maps $\varphi_{H_1+1}, \ldots, \varphi_T$. Write
$\Pi_h = \varphi_{H_1+h} \circ \cdots \circ \varphi_{H_1+1}$ for the first $h$
days of the operating phase and $\Pi = \Pi_{H_2}$ for the whole phase.
```

### M-A-38: Theorem thm:a_loop (perpetual safety) with proof and scope
- From: Loop Constraint (sec:loop).
- Kind: proof.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: nothing.

```latex
\begin{theorem}[Perpetual safety]
\label{thm:a_loop}
Let a solution satisfy every row of the problem. Then, for every cell and every day
$k \geq 1$ of the endless schedule, the true statistics $\theta_k$ and the reference
values $\zeta_k$ are at most a tracked value of the horizon that lies in
$\mathcal{C}_{i\ell}$. Therefore $\zeta_k \in \mathcal{C}_{i\ell}$ and
$\Pr(D_{i\ell k} > \tau_{i\ell}) \leq \varepsilon_{i\ell}$ on every day $k \geq 1$.
\end{theorem}
\begin{proof}
Fix a cell. Let $\xi_k$ be its tracked full state, and $\theta_k$ and $\zeta_k$ the
true statistics and the reference trajectory of the endless schedule. By (M2) and
Lemma~\ref{lem:a_monotone}, the induction of Proposition~\ref{prop:a_domination}
gives $\theta_k \leq \zeta_k$ on every day $k \geq 1$.
\begin{itemize}[nosep,leftmargin=1.5em]
  \item Days $k \leq T$: $\zeta_k \leq \xi_k$ (Proposition~\ref{prop:a_domination})
        and $\xi_k \in \mathcal{C}_{i\ell}$ (consequence on the horizon,
        Section~\ref{sec:encoding}).
  \item By Proposition~\ref{prop:a_domination} and Lemma~\ref{lem:a_monotone},
        $\xi_{H_1+h} \geq \varphi_{H_1+h}(\xi_{H_1+h-1}) \geq \cdots \geq
        \Pi_h(\xi_{H_1})$ for $h = 1, \ldots, H_2$. With Eq.~\eqref{eq:a_loop},
        $\Pi(\xi_{H_1}) \leq \xi_T \leq \xi_{H_1}$.
  \item Let $\Pi^n(\xi_{H_1}) \leq \xi_{H_1}$ (true for $n = 0$). Then
        $\Pi^{n+1}(\xi_{H_1}) = \Pi(\Pi^n(\xi_{H_1})) \leq \Pi(\xi_{H_1}) \leq
        \xi_{H_1}$ by Lemma~\ref{lem:a_monotone}. So $\Pi^n(\xi_{H_1}) \leq
        \xi_{H_1}$ for every $n \geq 0$.
  \item Day $k = H_1 + n H_2 + h$ with $n \geq 0$, $1 \leq h \leq H_2$:
        \[
          \zeta_k = \Pi_h\bigl(\Pi^n(\zeta_{H_1})\bigr)
          \leq \Pi_h\bigl(\Pi^n(\xi_{H_1})\bigr)
          \leq \Pi_h(\xi_{H_1})
          \leq \xi_{H_1+h} \in \mathcal{C}_{i\ell}.
        \]
\end{itemize}
Therefore, on every day $k \geq 1$, $\theta_k \leq \zeta_k \leq s$ for a tracked value
$s \in \mathcal{C}_{i\ell}$. Since $\mathcal{C}_{i\ell}$ is a down-set,
$\zeta_k \in \mathcal{C}_{i\ell}$, and (M3) gives $\Pr(D_{i\ell k} > \tau_{i\ell})
\leq \varepsilon_{i\ell}$.
\end{proof}

The theorem holds for every model and both repair laws, because it uses only
the facts (M1)--(M3) and Lemma~\ref{lem:a_monotone}. It holds in both encodings and
in every reliability form. It needs no relation between $H_1$ and $H_2$: the
transitory phase may be shorter or longer than the operating phase, because
$H_1$ only marks the day at which the repeated block starts. Slack in the tracked
values does not weaken it: the proof compares the reference trajectory with the
tracked values, whatever slack they carry.
```

### M-A-39: Why the loop row covers the full state, with an example
- From: Loop Constraint (sec:loop), remark.
- Kind: remark.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: nothing.

```latex
\emph{Remark (why the loop row covers the full state).} The third step of the proof
needs $\Pi(\xi_{H_1}) \leq \xi_{H_1}$ in every coordinate, anchors included.
Example: law $\ARDone$, $\rho = 0.5$, the first phase ends at $(\mu, g) = (0.4,
0.1)$, and the operating phase is a mission ($+0.1$), a repair, a mission
($+0.05$). The phase ends at $(0.35, 0.3)$, since the repair gives $0.5 \cdot 0.5 +
0.5 \cdot 0.1 = 0.3$. The row on the mean holds ($0.35 \leq 0.4$); the row on the
anchor fails ($0.3 > 0.1$). Each repetition raises the state after the repair by
$0.075$, so the mean grows without bound: the third repetition already peaks at
$0.525$, above the checked peak $0.5$. A loop row on the mean alone would accept
this unsafe schedule.
```

### M-A-40: No loop row on a scalar bound, with an example
- From: Loop Constraint (sec:loop), remark.
- Kind: remark.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: one sentence: the row compares each coordinate, never a scalar function of them.

```latex
\emph{Remark (no loop row on a scalar bound).} The loop row must compare the tracked
quantities one by one. A single comparison of a scalar function of them, such as the
failure-probability bound $U(\mu, v)$, does not chain: $U$ takes the same value at
many states, and its level sets are not invariant under the day maps. Example
(Cantelli, $U = v / (v + (\tau - \mu)^2)$, $\tau = 10$, $\rho = 0.5$, law
$\ARDinf$; the operating phase is one repair day followed by mission days that add
$3$ to the mean and $3.99$ to the variance in total): from $(\mu, v) = (4, 9)$,
$U = 0.2000$ at the end of the first phase and $U = 0.1997$ at the end of the
operating phase, so a scalar row would accept. The next repetition gives $U =
0.2151 > 0.2$, and $U$ tends to $0.2495$. Eq.~\eqref{eq:a_loop} rejects the
schedule, since the mean rises from $4$ to $5$. The loop row states an order between
tracked quantities; it is not a stochastic order between damage laws.
```

### M-A-41: Proposition prop:a_ard1_replacement with proof and consequences
- From: Loop Constraint (sec:loop).
- Kind: proof.
- Suggested destination: Overview of the problem: Stochastic loop constraint (with Stochastic orderings: loop constraint for each degradation model).
- Spec keeps: one-sentence notes: an ARD1 cell with rho < 1 that gains mean damage in the operating phase must be replaced in that phase; with allow_replacement = false its train cannot fly such a mission there; Wiener likewise (pointer).

```latex
\begin{proposition}[Replacement under $\ARDone$]
\label{prop:a_ard1_replacement}
Let a cell follow $\ARDone$ with $\rho_{i\ell} < 1$, and let its train fly, in the
operating phase, at least one mission with a positive mean increment for the cell.
Then every feasible solution has at least one replacement of the cell in the
operating phase.
\end{proposition}
\begin{proof}
Let the cell have no replacement in the operating phase. By the anchor rows,
$g_k = g_{k-1}$ on a day without intervention and $g_k = \mu_k$ on a repair day.
Every day satisfies $\mu_k \geq g_k$: on an intervention day with equality, and on
other days because $\mu$ does not decrease and $g$ stays constant (with $g_0 \leq
\mu_0$, Section~\ref{sec:repair_laws}). On a repair day $k$ of the operating phase,
\[
  g_k = \mu_k \geq (1-\rho)\,\mu_{k-1} + \rho\, g_{k-1}
  = g_{k-1} + (1-\rho)\,(\mu_{k-1} - g_{k-1}).
\]
So $g_T \geq g_{H_1} + (1-\rho) \sum (\mu_{k-1} - g_{k-1})$, the sum over the
repair days of the operating phase, each term non-negative. The anchor row $g_T
\leq g_{H_1}$ and $\rho < 1$ force every term to $0$. Then every repair day gives
$\mu_k \geq \mu_{k-1}$, every other day gives $\mu_k \geq \mu_{k-1} + \delta\mu_k$,
and
\[
  \mu_T \geq \mu_{H_1} + \textstyle\sum_{k = H_1+1}^{T} \delta\mu_k > \mu_{H_1},
\]
which violates the row $\mu_T \leq \mu_{H_1}$. Therefore the cell is replaced at
least once in the operating phase.
\end{proof}

The requirement is not an artefact of the rows. Without a replacement, each repair
leaves the share $1-\rho$ of the damage since the last intervention in place, so the
reference mean grows without bound over the repetitions, as in the first remark
above. With $\rho = 1$ a repair returns the state to the anchor, and no replacement
is needed. With \texttt{allow\_replacement = false}, the train of such a cell cannot
fly, in the operating phase, a mission with a positive increment for the cell. If the
fleet cannot cover the missions without such a flight, the model is infeasible. The Wiener model needs a replacement in the operating
phase for another reason, its variance (Section~\ref{sec:dyn_wiener}).
```
