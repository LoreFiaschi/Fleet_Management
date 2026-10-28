# Moved from fragments_C.tex (writer C, 26 Oct 2026)

Blocks removed from the inputs, outputs and schema sections when they were rewritten as a pure specification (conventions.md, "Spec versus scoping"). The LaTeX of each block is copied verbatim from the version before the rewrite. Labels in the verbatim text (for example `eq:c_step`, `eq:c_phases`) may no longer exist in the spec.

### M-C-1: Range of epsilon and the variance range V_max
- From: Input Specification, Reliability (`sec:input_reliability`)
- Kind: remark
- Suggested destination: Overview of the problem (Abstract problem formulation)
- Spec keeps: the key row `epsilon` and the check `epsilon < 0.5` at G and W cells

```latex
\emph{Remark (range of $\varepsilon$).}
The Gaussian and Wiener rows contain $\bigl(\Phi^{-1}(1-\varepsilon)\bigr)^2$.
This number is positive only for $\varepsilon < 0.5$.
The variance range $V_{\max} = \tau^2/\bigl(\Phi^{-1}(1-\varepsilon)\bigr)^2$ is the largest variance that the row admits.
It is the natural range of the variance, and the big-M constant of the variance rows (Section~\ref{sec:encoding}).
The ratio $V_{\max}/\tau^2$ is $0.18$ at $\varepsilon = 0.01$, $0.61$ at $0.1$, $3.6$ at $0.3$ and about $1600$ at $0.49$.
The constant is therefore moderate unless $\varepsilon$ is close to $0.5$.
The other models and bounds have no limit below $\varepsilon = 1$: for example, the variance cap $\tau^2\varepsilon/(1-\varepsilon)$ of the Cantelli row is finite for every $\varepsilon < 1$.
```

### M-C-2: Starting anchor: meaning, default and worked example
- From: Input Specification, Repair and Starting Anchors (`sec:input_repair`)
- Kind: rationale
- Suggested destination: Maintenance strategies (connection to the fleet management formulation)
- Spec keeps: the table rows; the default of each starting anchor is the initial value of the same state

```latex
Under the law \texttt{ard1}, a repair removes the fraction $\rho$ of the level accumulated since the last intervention (Section~\ref{sec:repair_laws}).
Before the first intervention, the reference level is the starting anchor of the state.
The default of each starting anchor is the initial value of the same state: $\mu_{0}$, $v_{0}$, $R_{0}$ and $K_{0}$ (Section~\ref{sec:input_initial}).
The largest allowed value is this default, because damage only grows between interventions (Section~\ref{sec:consistency_checks}).
A larger anchor gives a weaker repair and a higher state, so the default never overstates what a repair achieves.
For example, let $\mu_0 = 0.5$ and $\rho = 0.5$, and let a repair take place on day~1 before any mission.
With the starting anchor $0$ the mean falls to $0.25$; with the default anchor $0.5$ it stays at $0.5$.
The input is exact when the maintenance history of the component is known.
```

### M-C-3: Known and random initial damage: why the defaults
- From: Input Specification, Initial State (`sec:input_initial`)
- Kind: rationale
- Suggested destination: Literature review (Stochastic degradation and fatigue models: conservative surrogates for the empirical damage-increment distribution)
- Spec keeps: the table rows with their defaults

```latex
The defaults describe a known initial damage: the mean $\mu_0$ with no spread, so that $v_0 = b^0 = R_0 = 0$ and $K_0 = s\,\mu_0$.
The optional keys describe a random initial damage.
Section~\ref{sec:dyn_rainflow} states why these values keep the certificate valid.
```

### M-C-4: Replacement inputs: pointers to the rows and the reset of R and K
- From: Input Specification, Replacement (`sec:input_replacement`)
- Kind: remark
- Suggested destination: Overview of the problem (Damage dynamics)
- Spec keeps: the table rows

```latex
The state after a replacement comes from these keys for every model.
The defaults give a part in the zero state.
Section~\ref{sec:replacement} states the replacement rows; Section~\ref{sec:dyn_rainflow} states the reset of $R$ and $K$.
For Gamma, the rate of the state after a replacement is \texttt{gamma\_beta\_new} (Section~\ref{sec:input_constants}).
```

### M-C-5: The operating phase repeats; index k = 0
- From: Input Specification, Sizes and Horizon (`sec:input_scalars`)
- Kind: rationale
- Suggested destination: Overview of the problem (Stochastic loop constraint)
- Spec keeps: `T = H_1 + H_2` and the definition of the two phases

```latex
The operating phase is meant to repeat without end; the loop constraint of Section~\ref{sec:loop} makes every repetition safe.

Index $k = 0$ holds the initial state.
```

### M-C-6: Shape ambiguity: swapped mission and component axes
- From: Input Specification, Increment Profiles (`sec:input_increments`)
- Kind: remark
- Suggested destination: Implementation details
- Spec keeps: the rule that a shape fitting both (L, M, H) and (F, L, M) is read as (L, M, H)

```latex
The parser cannot detect an array whose mission axis and component axis are swapped when $M = L$.
```

### M-C-7: Phase lookup: worked examples
- From: Input Specification, Increment Profiles (`sec:input_increments`)
- Kind: example
- Suggested destination: Overview of the problem (Damage dynamics)
- Spec keeps: Equation `eq:c_phase` and the rule that each profile key is looked up on its own

```latex
For a scalar $H$ without transitory profile, both phases read the same $H$ entries, one after the other.
For $H = [2, 3]$ without transitory profile, days $1,\ldots,5$ read the operating entries $1, 2, 1, 2, 3$.
With a transitory profile, days $1, 2$ read its entries $1, 2$ and days $3, 4, 5$ read the operating entries $1, 2, 3$.
```

### M-C-8: Meaning of `support`: one width in every bound
- From: Input Specification, Increment Profiles (`sec:input_increments`)
- Kind: remark
- Suggested destination: Literature review (Stochastic degradation and fatigue models: conservative surrogates for the empirical damage-increment distribution)
- Spec keeps: the key row `support`: the width of the range of the increment

```latex
The key \texttt{support} keeps one meaning in every bound: the width of the range of the increment.
A width of $0$ means that the increment is known.
```

### M-C-9: Smallest Gamma rate of a cell: pointer
- From: Input Specification, Model Constants (`sec:input_constants`)
- Kind: remark
- Suggested destination: Stochastic orderings (within each degradation family)
- Spec keeps: the rows `gamma_beta_0` and `gamma_beta_new` state that an omitted rate does not enter the smallest rate

```latex

For Gamma, the formulation computes the smallest rate of each cell from the rates of the terms with a positive mean: the entries of \texttt{gamma\_beta} and \texttt{gamma\_beta\_trans}, and the rates \texttt{gamma\_beta\_0} and \texttt{gamma\_beta\_new} where they are given (Section~\ref{sec:dyn_gamma}).
No input gives this rate.
```

### M-C-10: Costs: pointer to the objective
- From: Input Specification, Costs and Objective (`sec:input_costs`)
- Kind: remark
- Suggested destination: Overview of the problem (Cost function)
- Spec keeps: nothing

```latex
All costs are finite.
The objective and its three modes (\texttt{objective\_mode}, Section~\ref{sec:run_options}) are stated in Section~\ref{sec:objective}.
```

### M-C-11: Run options: scope and sweeps
- From: Input Specification, Run Options (`sec:run_options`)
- Kind: remark
- Suggested destination: Implementation details
- Spec keeps: the sentence that no input key selects a warm start from a previous result

```latex
Run options are keys of the input.
They do not change the mathematical problem, except \texttt{objective\_mode} and \texttt{evaluation\_horizon}, which choose the objective.

A horizon sweep is not an input option; the sweep functions of Section~\ref{sec:sweeps} read the same input and solve it for several horizons.
```

### M-C-12: Rejected keys: why the table exists
- From: Input Specification, Rejected Keys (`sec:input_rejected`)
- Kind: rationale
- Suggested destination: Implementation details
- Spec keeps: the rule that a rejected key raises an error with the message of the table

```latex
Each key of the table below names a quantity that has another name, or another meaning, in this specification.
```

### M-C-13: Key-table rows: removed wording (sizes, selectors, reliability, initial state, replacement)
- From: Input Specification, tables of the subsections Sizes and Horizon to Replacement
- Kind: remark
- Suggested destination: Implementation details
- Spec keeps: each row, shortened to key, shape, default, cells and a one-line meaning (the ranges moved to the checks)

```latex
\keyrow{\texttt{F}}{integer}{required}{all}{Number of trains. Must exceed $M$}
\keyrow{\texttt{M}}{integer}{required}{all}{Number of missions. Every mission is flown by one train on every day}
\keyrow{\texttt{L}}{integer}{inferred}{all}{Number of components per train. When \texttt{model} is a list or an array, $L$ is the length of its component axis, and a given \texttt{L} must equal it. Required when \texttt{model} is one string and $L > 1$}
\keyrow{\texttt{H}}{integer or \texttt{[H1, H2]}}{required}{all}{Horizon (below). A scalar $H$ means $H_1 = H_2 = H$}

\keyrow{\texttt{model}}{selector}{required}{all}{Degradation model of the cell: \texttt{gaussian}, \texttt{inverse\_gaussian}, \texttt{wiener}, \texttt{gamma} or \texttt{rainflow}}
\keyrow{\texttt{repair\_model}}{selector}{\texttt{ardinf}}{all}{Repair law of the cell (Section~\ref{sec:repair_laws}): \texttt{ardinf} (the repair reduces the whole level) or \texttt{ard1} (it reduces the level accumulated since the last intervention). Every model accepts both}
\keyrow{\texttt{bound\_method}}{selector}{\texttt{cantelli}}{RF}{Tail bound of the reliability row (Section~\ref{sec:dyn_rainflow}): \texttt{markov}, \texttt{cantelli}, \texttt{hoeffding}, \texttt{bernstein} or \texttt{chernoff}}
\keyrow{\texttt{reliability\_impl}}{selector}{\texttt{exact}}{G, W, RF (Ca, Be, Ho)}{Form of the reliability row (Section~\ref{sec:linear_forms}): \texttt{exact}, \texttt{tangent} or \texttt{pwl}. A cell whose row is linear (IG, Ga, Mk, Ch) is built in the form \texttt{exact}, and the value is ignored}
\keyrow{\texttt{encoding}}{selector}{\texttt{bigm}}{all}{Encoding of the dynamics rows (Section~\ref{sec:encoding}): \texttt{bigm} or \texttt{indicator}}
\keyrow{\texttt{component\_names}}{list of $L$ strings}{\texttt{Component} $n$ for component $n$}{all}{Name of each component, shown by the plots. A name is not empty after blanks are removed}

\keyrow{\texttt{tau}}{cell value}{required}{all}{Degradation threshold $\tau_{i\ell} > 0$}
\keyrow{\texttt{epsilon}}{cell value}{required}{all}{Reliability level $\varepsilon_{i\ell} \in (0,1)$: the probability that the degradation exceeds $\tau_{i\ell}$ must not exceed it on any day. At G and W cells $\varepsilon_{i\ell} < 0.5$}
\keyrow{\texttt{tangent\_ref}}{cell value}{$0.5$}{cells with form \texttt{tangent}}{Relative tangent point in $[0,1)$: the tangent is taken at the mean $\mu_r = \texttt{tangent\_ref}\cdot\tau$}
\keyrow{\texttt{pwl\_points}}{integer}{$8$}{cells with form \texttt{pwl}}{Number of segments of the piecewise linear form}

\keyrow{\texttt{mu\_0}}{cell value}{required}{all}{Initial mean degradation $\mu_{0,i\ell}$, with $0 \le \mu_{0,i\ell} < \tau_{i\ell}$}
\keyrow{\texttt{v\_0}}{cell value}{$0$}{G, W, RF (Ca, Be)}{Initial variance $v_{0,i\ell} \ge 0$. The value $0$ means a known initial damage}
\keyrow{\texttt{support\_0}}{cell value}{$0$}{RF (Be)}{Width $b^{0}_{i\ell}$ of the range of the initial damage. The value $0$ means a known initial damage}
\keyrow{\texttt{R\_0}}{cell value}{$0$}{RF (Ho)}{Initial value $R_{0,i\ell}$ of the Hoeffding budget: the square of the width of the range of the initial damage}
\keyrow{\texttt{cgf\_0}}{cell value}{$s\,\mu_0$}{RF (Ch)}{Initial value $K_{0,i\ell}$ of the Chernoff quantity: the cumulant generating function of the initial damage at $s_{i\ell} = \texttt{s\_chernoff}$. It is at least $s\,\mu_0$}

\keyrow{\texttt{replacement\_mu}}{cell value}{$0$}{all}{Mean degradation $\mu^{\new}_{i\ell}$ after a replacement}
\keyrow{\texttt{replacement\_v}}{cell value}{$0$}{G, W, RF (Ca, Be)}{Variance $v^{\new}_{i\ell}$ after a replacement}
\keyrow{\texttt{support\_new}}{cell value}{$0$}{RF (Be)}{Width $b^{\new}_{i\ell}$ of the range of the damage after a replacement}
\keyrow{\texttt{R\_new}}{cell value}{$0$}{RF (Ho)}{Hoeffding budget $R^{\new}_{i\ell}$ after a replacement}
\keyrow{\texttt{cgf\_new}}{cell value}{$s\,\mu^{\new}$}{RF (Ch)}{Chernoff quantity $K^{\new}_{i\ell}$ after a replacement; at least $s\,\mu^{\new}$}
\keyrow{\texttt{replacement\_as\_new}}{flag}{\texttt{false}}{all}{If \texttt{true}, a replacement gives a component in the zero state, and every replacement input of this table must be $0$}
\keyrow{\texttt{allow\_replacement}}{flag}{\texttt{true}}{all}{If \texttt{false}, no replacement is possible: the variables $x^r$ and their rows are not built}
```

### M-C-14: Key-table rows: removed wording (profiles, constants, costs, run options)
- From: Input Specification, tables of the subsections Increment Profiles to Run Options
- Kind: remark
- Suggested destination: Implementation details
- Spec keeps: each row, shortened to key, shape, default, cells and a one-line meaning (the ranges moved to the checks)

```latex
\keyrow{\texttt{mu}}{profile}{required}{all}{Mean $\mu^{\inc}_{i\ell jk} \ge 0$ of the increment. For Ga the increment has shape $\mu\,\beta$ and rate $\beta$, so that its mean is $\mu$}
\keyrow{\texttt{v}}{profile}{required}{G, RF (Ca, Be)}{Variance $v^{\inc}_{i\ell jk} \ge 0$ of the increment}
\keyrow{\texttt{support}}{profile}{required}{RF (Ho, Be)}{Width $b^{\inc}_{i\ell jk} \ge 0$ of the range of the increment: the increment lies in an interval of this length}
\keyrow{\texttt{cgf}}{profile}{required}{RF (Ch)}{Cumulant generating function $K^{\inc}_{i\ell jk} = \ln \mathbb{E}\,e^{s X}$ of the increment $X$ at $s = s_{i\ell}$}
\keyrow{\texttt{s\_chernoff}}{cell value}{required}{RF (Ch)}{Chernoff parameter $s_{i\ell}$, with $s_{i\ell} > \ln(1/\varepsilon_{i\ell})/\tau_{i\ell}$}
\keyrow{\texttt{gamma\_beta}}{scalar, $(L)$, $(F, L)$, $(F, L, M)$ or $(F, L, M, H_2)$}{required}{Ga}{Rate $\beta_{i\ell jk} > 0$ of the Gamma increment. A shape without a day axis is repeated along the days}
\keyrow{\texttt{mu\_trans}, \texttt{v\_trans}, \texttt{support\_trans}, \texttt{cgf\_trans}}{transitory profile}{none}{as the key without the suffix}{Increment parameters of the transitory phase. A transitory profile requires its operating profile}
\keyrow{\texttt{gamma\_beta\_trans}}{scalar, $(L)$, $(F, L)$, $(F, L, M)$ or $(F, L, M, H_1)$}{none}{Ga}{Rates of the transitory phase}

\keyrow{\texttt{sigma}}{cell value}{required}{W}{Diffusion constant $\sigma_{i\ell} > 0$: every mission adds $\sigma_{i\ell}^2$ to the variance}
\keyrow{\texttt{eta}}{cell value}{required}{IG}{Dispersion $\eta_{i\ell} > 0$ of the inverse Gaussian law of each term (Section~\ref{sec:dyn_ig})}
\keyrow{\texttt{v\_max\_user}}{cell value}{none}{G, W, RF (Ca, Be)}{Upper bound $V^{\mathrm{user}}_{\max,i\ell} > 0$ on the variance. If it is given, the row $v_{i\ell k} \le V^{\mathrm{user}}_{\max,i\ell}$ holds on every day. If it is not given, only the reliability row bounds the variance}
\keyrow{\texttt{gamma\_beta\_0}}{cell value}{none (known damage)}{Ga}{Rate of the initial damage, which is a Gamma variable with mean $\mu_0$. If it is omitted, the initial damage is the known constant $\mu_0$, and it does not enter the smallest rate}
\keyrow{\texttt{gamma\_beta\_new}}{cell value}{none (known damage)}{Ga}{Rate of the damage after a replacement, which is a Gamma variable with mean $\mu^{\new}$. If it is omitted, the damage after a replacement is the known constant $\mu^{\new}$, and it does not enter the smallest rate}

\keyrow{\texttt{C\_M}}{scalar}{required}{all}{Cost $C_M \ge 0$ of one depot day of one train, whatever the number of jobs done on that day}
\keyrow{\texttt{C\_R}}{cell value}{required}{all}{Repair cost $C_{R,i\ell} \ge 0$ per unit of degradation removed. At a cell with $C_{R,i\ell} = 0$ the variable $z$ and its rows are not built}
\keyrow{\texttt{C\_rep}}{cell value}{required}{all}{Replacement cost $C_{\mathrm{rep},i\ell} \ge 0$}
\keyrow{\texttt{C\_D}}{scalar}{required}{all}{Damage regularisation coefficient $C_D \ge 0$. If $C_D = 0$, the variable $u$ and its rows are not built, and \texttt{penalty\_type} is ignored}
\keyrow{\texttt{penalty\_type}}{string}{\texttt{inf\_norm}}{all}{Form of the damage regularisation (Section~\ref{sec:damage_reg}): \texttt{inf\_norm} or \texttt{quadratic}. If $C_D > 0$, the value \texttt{quadratic} requires \texttt{reliability\_impl} \texttt{exact} at every cell with a quadratic reliability row (G, W, RF with Ca, Ho or Be); another value is an error}

\optrow{\texttt{verbose}}{integer}{$1$}{Gurobi output flag: $0$ for no log, $1$ for the log}
\optrow{\texttt{mip\_gap}}{number $\ge 0$}{$0.1$}{Relative gap between the best solution and the best bound at which Gurobi stops (the MIP gap; MIP means mixed-integer program). With the default, the status \texttt{optimal} means that the best solution is at most $10\,\%$ above the best bound. The gap reached is reported as \texttt{mip\_gap} in the result}
\optrow{\texttt{time\_limit}}{number $> 0$}{$3600$}{Time limit of Gurobi, in seconds}
\optrow{\texttt{fast}}{flag}{\texttt{false}}{If \texttt{true}, a preset of Gurobi parameters that favours good solutions found early: \texttt{MIPFocus} $= 1$, \texttt{Heuristics} $= 0.5$, \texttt{ImproveStartGap} $= 0.5$ and \texttt{NoRelHeurTime} $= \max(2,\, 0.15\,\texttt{time\_limit})$}
\optrow{\texttt{gurobi\_params}}{mapping}{none}{Gurobi parameters, by name. They are set after every other option, and so override it}
\optrow{\texttt{objective\_mode}}{string}{\texttt{total}}{\texttt{total}: the cost of all $T$ days; \texttt{operating\_average}: the cost per day of the operating phase; \texttt{evaluation\_total}: the projected cost over \texttt{evaluation\_horizon} days. Each mode adds the damage penalty $C_D\,u$ once (Section~\ref{sec:objective})}
\optrow{\texttt{evaluation\_horizon}}{integer}{none}{Evaluation horizon $E$. Required, and used, only if \texttt{objective\_mode} is \texttt{evaluation\_total}. $E > H_1$ whenever it is given. If it is given in another mode, the result reports the projected cost for it (Section~\ref{sec:output})}
\optrow{\texttt{progress\_interval\_seconds}}{number $> 0$}{$5.0$}{Minimum time in seconds between two records of the progress report (Section~\ref{sec:output_reports}). A record is also made whenever the number of solutions found changes}
\optrow{\texttt{relaxation\_warm\_start}}{flag}{\texttt{false}}{If \texttt{true}, and no other warm start is applied, a relaxation start is computed before the solve (Section~\ref{sec:warm_starts})}
\optrow{\texttt{relaxation\_time\_limit}}{number $> 0$}{$60.0$}{Time limit in seconds of the relaxation start}
\optrow{\texttt{validation}}{mapping}{none}{Settings of the Gamma validators (Section~\ref{sec:validation_gamma}): \texttt{mode}, \texttt{repetitions}, \texttt{random\_seed}, \texttt{maximum\_schedule\_failure\_rate}, \texttt{confidence\_level} and \texttt{batch\_size}. \texttt{solve} does not read it}
```

### M-C-15: Checks: introduction, keys, shapes and entries (rules with reasons)
- From: Input Specification, Consistency Checks (`sec:consistency_checks`)
- Kind: rationale
- Suggested destination: Implementation details
- Spec keeps: the rules only

```latex
The solver must validate all of the following before it builds the model.
A violation raises a \texttt{ValueError}.
Its message names the key and, for a cell value, the cell $(i,\ell)$.
A check of one model runs only at the cells of that model, on the entries where the key applies.
Each item gives the rule and, after the dash, the reason for it.

\textbf{Keys, shapes and entries.}
\begin{itemize}[leftmargin=1.5em]
  \item \emph{Keys.} Every key is in the tables of Section~\ref{sec:inputs}. The keys \texttt{F}, \texttt{M}, \texttt{H}, \texttt{model}, \texttt{tau}, \texttt{epsilon}, \texttt{rho}, \texttt{mu\_0}, \texttt{mu}, \texttt{C\_M}, \texttt{C\_R}, \texttt{C\_rep} and \texttt{C\_D} are present, and so are the keys that each cell type requires (Section~\ref{sec:schema}). --- A misspelled or missing key must not change the problem without a message.
  \item \emph{Shapes.} Each value has one of the shapes of its table row. The last axis of an operating profile has $H_2$ entries, and that of a transitory profile has $H_1$ entries. \texttt{component\_names} has $L$ entries. --- Silent broadcasting by the array library could read a value for another cell.
  \item \emph{Entries.} At a cell where a key applies, every entry is a finite number, or a string for a selector; \texttt{null} is an error. --- A missing value at a cell must not take a default silently.
\end{itemize}
```

### M-C-16: Checks: sizes and selectors (rules with reasons)
- From: Consistency Checks
- Kind: rationale
- Suggested destination: Overview of the problem (Scheduling and maintenance variables)
- Spec keeps: the rules only

```latex
\textbf{Sizes and selectors.}
\begin{itemize}[leftmargin=1.5em]
  \item $F$, $M$ and $L$ are positive integers, and $F > M$. --- Every mission needs a train on every day; with $F = M$ no train could be in the depot, and no repair could take place.
  \item $H$ is an integer $\ge 1$, or a pair of integers $H_1, H_2 \ge 1$. --- Each phase has at least one day.
  \item \texttt{model} entries are \texttt{gaussian}, \texttt{inverse\_gaussian}, \texttt{wiener}, \texttt{gamma} or \texttt{rainflow}. \texttt{repair\_model} entries are \texttt{ard1} or \texttt{ardinf}, for every model. --- Another value names no model or no repair law.
  \item \texttt{bound\_method} entries at rainflow cells are \texttt{markov}, \texttt{cantelli}, \texttt{hoeffding}, \texttt{bernstein} or \texttt{chernoff}. --- A rainflow cell without one of these bounds has no reliability row.
  \item Every non-\texttt{null} entry of \texttt{reliability\_impl} is \texttt{exact}, \texttt{tangent} or \texttt{pwl}, also at a cell whose row is linear (IG, Ga, Mk, Ch); there the value is ignored, and the form \texttt{exact} is built. \texttt{encoding} entries are \texttt{bigm} or \texttt{indicator}. --- An unknown value must raise an error at every cell; reading it as \texttt{exact} would turn a linear model into a nonconvex one without a message.
  \item \texttt{penalty\_type} is \texttt{inf\_norm} or \texttt{quadratic}. If $C_D > 0$ and \texttt{penalty\_type} is \texttt{quadratic}, \texttt{reliability\_impl} is \texttt{exact} at every cell with a quadratic reliability row (G, W, and RF with Ca, Ho or Be); another value is an error. --- The quadratic penalty is a quadratic row, and the linear forms exist to give a linear model.
\end{itemize}
```

### M-C-17: Checks: reliability, repair and initial state (rules with reasons)
- From: Consistency Checks
- Kind: rationale
- Suggested destination: Overview of the problem (Damage dynamics)
- Spec keeps: the rules only

```latex
\textbf{Reliability, repair and initial state.}
\begin{itemize}[leftmargin=1.5em]
  \item $\tau_{i\ell} > 0$, and finite. --- The threshold is a positive degradation level.
  \item $\varepsilon_{i\ell} \in (0, 1)$, and $\varepsilon_{i\ell} < 0.5$ at G and W cells. --- A probability lies in $(0,1)$; the squared normal quantile of the Gaussian and Wiener rows is positive only below $0.5$ (Section~\ref{sec:input_reliability}).
  \item $\rho_{i\ell} \in (0, 1]$. --- A repair removes a positive fraction of the level, at most all of it.
  \item $0 \le \mu_{0,i\ell} < \tau_{i\ell}$. --- A component at its threshold has failed already.
  \item $v_0$, \texttt{support\_0} and $R_0$ are $\ge 0$; if \texttt{allow\_replacement} is \texttt{true}, so are \texttt{replacement\_mu}, \texttt{replacement\_v}, \texttt{support\_new} and $R^{\new}$. --- They describe a damage, a variance, a width or a squared width.
  \item \texttt{tangent\_ref} $\in [0, 1)$ at cells with form \texttt{tangent}, and \texttt{pwl\_points} is an integer $\ge 1$. --- At $\texttt{tangent\_ref} = 1$ the tangent row reduces to $Q \le 0$, and a value outside $[0,1]$ has no tangent point in $[0,\tau]$.
  \item Starting anchors: $0 \le \texttt{anchor\_mu\_0} \le \mu_0$, $0 \le \texttt{anchor\_v\_0} \le v_0$, $0 \le \texttt{anchor\_R\_0} \le R_0$ and $0 \le \texttt{anchor\_K\_0} \le K_0$, at the cells that read them. --- Damage only grows between interventions, so the level after the last intervention cannot exceed the initial level.
  \item If \texttt{replacement\_as\_new} is \texttt{true}, then \texttt{replacement\_mu}, \texttt{replacement\_v}, \texttt{support\_new}, \texttt{R\_new} and \texttt{cgf\_new} are $0$ at every cell that reads them. The check does not run when the flag is \texttt{false}. --- The shortcut means a new part, and a non-zero value would be discarded without a message.
\end{itemize}
```

### M-C-18: Checks: natural ranges and replacement states (rules with reasons)
- From: Consistency Checks
- Kind: rationale
- Suggested destination: Overview of the problem (Abstract problem formulation)
- Spec keeps: the rules only

```latex
\textbf{Natural ranges and replacement states.}
\begin{itemize}[leftmargin=1.5em]
  \item Every initial value ($\mu_0$, $v_0$, $R_0$, $K_0$) and every replacement value ($\mu^{\new}$, $v^{\new}$, $R^{\new}$, $K^{\new}$, checked only if \texttt{allow\_replacement} is \texttt{true}) lies in $[0, M_s]$, where $M_s$ is the natural range of its state: $\tau$ for a mean; $V_{\max} = \tau^2/\bigl(\Phi^{-1}(1-\varepsilon)\bigr)^2$ for the variance at G and W cells; $\varepsilon\tau^2/(1-\varepsilon)$ for the variance at Ca cells; $\tau(\tau - 2c)/(2\kappa)$ at Be cells, with $c = \kappa b/3$ and $\kappa = \ln(1/\varepsilon)$; $2\tau^2/\kappa$ for $R$ at Ho cells; $s\tau - \kappa$ for $K$ at Ch cells (Sections~\ref{sec:dyn_gaussian} to~\ref{sec:dyn_rainflow}). A starting anchor lies between $0$ and the initial value of its state, so it lies in the range as well. --- The big-M rows take the natural range as their constant, so a value outside it violates a row at once.
  \item If \texttt{allow\_replacement} is \texttt{true}, the replacement state $(\mu^{\new}, v^{\new}, R^{\new}, K^{\new})$ of every cell satisfies every row that holds on a replacement day, in the form built at the cell: the reliability row in the form of \texttt{reliability\_impl} (for \texttt{pwl}, the cap of the segment that contains $\mu^{\new}$), its side row, and $v^{\new} \le V^{\mathrm{user}}_{\max,i\ell}$ where \texttt{v\_max\_user} is given (Sections~\ref{sec:dyn_gaussian} to~\ref{sec:dyn_rainflow} and~\ref{sec:linear_forms}). --- The rows hold on the day of a replacement; otherwise a replacement is never feasible.
\end{itemize}
```

### M-C-19: Checks: increments (rules with reasons)
- From: Consistency Checks
- Kind: rationale
- Suggested destination: Overview of the problem (Damage dynamics)
- Spec keeps: the rules only

```latex
\textbf{Increments.}
\begin{itemize}[leftmargin=1.5em]
  \item $0 \le \mu^{\inc}_{i\ell jk} < \tau_{i\ell}$ for every mission $j$ and every day $k \in \{1,\ldots,T\}$. --- A mission with mean $\tau$ or more fails the component in one day; the mean $0$ is allowed, so that a mission that does not wear a component can be written as $0$.
  \item Where $\mu^{\inc}_{i\ell jk} = 0$, the values $v^{\inc}_{i\ell jk}$, $b^{\inc}_{i\ell jk}$ and $K^{\inc}_{i\ell jk}$ are $0$. --- A non-negative increment with mean $0$ is $0$ almost surely, so it has no variance, no range and a zero cumulant generating function.
  \item The values $v^{\inc}_{i\ell jk}$, $b^{\inc}_{i\ell jk}$ and $K^{\inc}_{i\ell jk}$ are $\ge 0$. A transitory profile has its operating profile. --- A variance and a width are not negative, and the lookup~\eqref{eq:c_phase} needs the operating entries after the transitory phase.
\end{itemize}
```

### M-C-20: Checks: Gaussian, inverse Gaussian, Wiener, Gamma and variance-cap cells (rules with reasons)
- From: Consistency Checks
- Kind: rationale
- Suggested destination: Stochastic orderings (within each degradation family)
- Spec keeps: the rules only

```latex
\textbf{Gaussian cells.}
\begin{itemize}[leftmargin=1.5em]
  \item $\mu_0 \ge 3\sqrt{v_0}$, $\mu^{\inc}_{i\ell jk} \ge 3\sqrt{v^{\inc}_{i\ell jk}}$ for every mission $j$ and day $k$, and, if \texttt{allow\_replacement} is \texttt{true}, $\mu^{\new} \ge 3\sqrt{v^{\new}}$. --- A Gaussian increment is then negative with probability at most about $0.13\,\%$, so the damage stays non-negative in practice.
\end{itemize}

\textbf{Inverse Gaussian cells.}
\begin{itemize}[leftmargin=1.5em]
  \item $\eta_{i\ell} > 0$. --- The dispersion of the inverse Gaussian law must be positive.
\end{itemize}

\textbf{Wiener cells.}
\begin{itemize}[leftmargin=1.5em]
  \item $\sigma_{i\ell} > 0$ and $v_{0,i\ell} \ge 0$. --- The diffusion constant is positive; a known initial damage has variance $0$ and is a valid input.
\end{itemize}

\textbf{Gamma cells.}
\begin{itemize}[leftmargin=1.5em]
  \item Every entry of \texttt{gamma\_beta} and of \texttt{gamma\_beta\_trans} is finite and $> 0$. --- The rate of a Gamma law is positive, and the smallest rate sets the cap (Section~\ref{sec:dyn_gamma}).
  \item \texttt{gamma\_beta\_0} and \texttt{gamma\_beta\_new}, where they are given, are finite and $> 0$. --- A rate is positive; an omitted rate means a known damage, which has no Gamma law and no rate.
\end{itemize}

\textbf{Cells with a variance cap.}
\begin{itemize}[leftmargin=1.5em]
  \item If \texttt{v\_max\_user} is given at a G, W or RF (Ca, Be) cell, then $V^{\mathrm{user}}_{\max,i\ell} > 0$ and $V^{\mathrm{user}}_{\max,i\ell} \ge v_{0,i\ell}$ and, if \texttt{allow\_replacement} is \texttt{true}, $V^{\mathrm{user}}_{\max,i\ell} \ge v^{\new}_{i\ell}$. --- A cap below the initial variance, or below the variance after a replacement, makes the problem infeasible.
\end{itemize}
```

### M-C-21: Checks: rainflow cells (rules with reasons)
- From: Consistency Checks
- Kind: rationale
- Suggested destination: Literature review (Stochastic degradation and fatigue models: conservative surrogates for the empirical damage-increment distribution)
- Spec keeps: the rules only; the name of the variance bound for a range of width w is dropped, the rule is `v <= b^2/4`

```latex
\textbf{Rainflow cells.}
\begin{itemize}[leftmargin=1.5em]
  \item Bernstein: $v^{\inc}_{i\ell jk} \le (b^{\inc}_{i\ell jk})^2/4$ for every mission $j$ and day $k$, $v_{0} \le (b^{0})^2/4$ and, if \texttt{allow\_replacement} is \texttt{true}, $v^{\new} \le (b^{\new})^2/4$. --- Popoviciu's inequality (below): a law on a range of width $w$ has variance at most $w^2/4$, so a larger variance contradicts the width.
  \item Bernstein: $\tau_{i\ell} > 2\kappa\, b_{i\ell}/3$, with $\kappa = \ln(1/\varepsilon_{i\ell})$ and $b_{i\ell}$ the support constant of Section~\ref{sec:dyn_rainflow}. --- Otherwise the certificate admits no positive variance: it is infeasible at zero damage, and at equality only $\mu = v = 0$ survives, so that the big-M rows break.
  \item Bernstein with the form \texttt{tangent}: $\texttt{tangent\_ref}\cdot\tau_{i\ell} < \sqrt{\tau_{i\ell}\,(\tau_{i\ell} - 2c_{i\ell})}$, with $c_{i\ell} = \kappa\, b_{i\ell}/3$. Bernstein with the form \texttt{pwl}: $\tau_{i\ell}/(2P) < \sqrt{\tau_{i\ell}\,(\tau_{i\ell} - 2c_{i\ell})}$, with $P = \texttt{pwl\_points}$. --- The row decreases in $\mu$, and at $\mu = 0$ it reads $v \le \bigl(\tau(\tau - 2c) - \mu_r^2\bigr)/(2\kappa)$ (Section~\ref{sec:linear_forms}), where $\mu_r$ is the tangent point ($\tau/(2P)$ for the first segment of \texttt{pwl}); otherwise no point with $v \ge 0$ satisfies the row, or only $\mu = v = 0$.
  \item Chernoff: \texttt{s\_chernoff} is given and $s_{i\ell} > \ln(1/\varepsilon_{i\ell})/\tau_{i\ell}$. --- The quantity $K$ is not negative, so the row $K - s\tau \le \ln\varepsilon$ needs $s\tau \ge \ln(1/\varepsilon)$; with a smaller $s$ every schedule is infeasible, and at equality only a component without damage is feasible.
  \item Chernoff: $K^{\inc}_{i\ell jk} \ge s_{i\ell}\,\mu^{\inc}_{i\ell jk}$ for every mission $j$ and day $k$, $K_{0} \ge s\,\mu_{0}$, at \texttt{ard1} cells $g^{K}_{0} \ge s\,g^{\mu}_{0}$, and, if \texttt{allow\_replacement} is \texttt{true}, $K^{\new} \ge s\,\mu^{\new}$. --- Jensen's inequality: the cumulant generating function of a random variable at $s > 0$ is at least $s$ times its mean; the two starting anchors describe the same damage after the last intervention, and the defaults satisfy the inequality.
\end{itemize}
```

### M-C-22: Checks: run options and costs (rules with reasons)
- From: Consistency Checks
- Kind: rationale
- Suggested destination: Overview of the problem (Cost function)
- Spec keeps: the rules only

```latex
\textbf{Run options.}
\begin{itemize}[leftmargin=1.5em]
  \item \texttt{objective\_mode} is \texttt{total}, \texttt{operating\_average} or \texttt{evaluation\_total}; in the last mode \texttt{evaluation\_horizon} is given. If \texttt{evaluation\_horizon} is given, in any mode, it is an integer $E > H_1$. --- The projected cost needs at least one operating day beyond the transitory phase; for a smaller $E$ the operating cost would have a negative weight.
  \item \texttt{mip\_gap} $\ge 0$; \texttt{time\_limit}, \texttt{progress\_interval\_seconds} and \texttt{relaxation\_time\_limit} are $> 0$; \texttt{gurobi\_params} and \texttt{validation} are mappings; \texttt{fast}, \texttt{relaxation\_warm\_start}, \texttt{allow\_replacement} and \texttt{replacement\_as\_new} are flags. --- A value outside its range must raise an error and must not be changed silently.
\end{itemize}

\textbf{Costs.}
\begin{itemize}[leftmargin=1.5em]
  \item $C_M$, $C_D$, $C_{R,i\ell}$ and $C_{\mathrm{rep},i\ell}$ are finite and $\ge 0$. --- A negative cost would reward a depot day, a repair, a replacement or damage.
\end{itemize}
```

### M-C-23: Variance of a law on a range of width w (proof and example)
- From: Consistency Checks, remark
- Kind: proof
- Suggested destination: Literature review (Stochastic degradation and fatigue models: conservative surrogates for the empirical damage-increment distribution)
- Spec keeps: the rule `v <= b^2/4` in the rainflow checks

```latex
\emph{Remark (Popoviciu's inequality).}
Let $X$ take values in an interval of width $w$, with midpoint $c$.
Then $\operatorname{Var} X \le \mathbb{E}\,(X - c)^2 \le (w/2)^2$.
The width is the only information on the range, and the range may start at any point $a \ge 0$.
The inequality $\operatorname{Var} X \le w^2/4$ holds for every such position, so it is a necessary condition for the pair (variance, width).
For example, increments in an interval of width $0.2$ have variance at most $0.01$; a two-point law on $\{0, 0.2\}$ with probability $1/2$ each has mean $0.1$ and reaches $0.01$.
A condition of the form $v \le \mu\,(b - \mu)$ assumes that the range starts at $0$, so it is not necessary for a width.
```

### M-C-24: Output: solutions and status (example)
- From: Output Specification, Rules for the Values (`sec:output`)
- Kind: example
- Suggested destination: Implementation details
- Spec keeps: the rule that arrays are null if no solution was found

```latex
The solution arrays are present whenever Gurobi has found at least one solution, and then \texttt{status} tells how the solve ended.
For example, a solution found before the time limit is reported with the status \texttt{time\_limit}.
If no solution was found, the solution arrays and the cost report are \texttt{null}, and \texttt{objective} and \texttt{mip\_gap} are \texttt{null}.
The four reports of the last block of the table are present in both cases.
```

### M-C-25: Output: tracked values (purpose and bounds)
- From: Output Specification, Rules for the Values
- Kind: remark
- Suggested destination: Implementation details
- Spec keeps: one sentence: the arrays hold the tracked variables and are upper bounds (Section `sec:encoding`)

```latex
\textbf{Tracked values.}
The arrays \texttt{mu}, \texttt{v}, \texttt{R}, \texttt{K}, \texttt{g\_mu}, \texttt{g\_v}, \texttt{g\_R} and \texttt{g\_K} hold the values of the tracked variables of the model, and \texttt{z} holds the repair-cost variable.
The result carries every tracked state and anchor, so that a validator can check every loop row of Section~\ref{sec:loop}.
The dynamics rows are one-sided (Section~\ref{sec:encoding}), so these values, and the costs computed from them, are upper bounds, not exact values.
```

### M-C-26: Output: repair cost on transitory days (explanation)
- From: Output Specification, Rules for the Values
- Kind: remark
- Suggested destination: Overview of the problem (Cost function)
- Spec keeps: Equation `eq:c_zpost` with its conditions

```latex
\textbf{Repair cost on transitory days.}
In the mode \texttt{operating\_average}, the variable $z$ is not built on the days of the transitory phase, because their cost does not enter the objective (Section~\ref{sec:repair_cost}).
For these days, at a cell with $C_{R,i\ell} > 0$, the result holds the value computed after the solve from the arrays \texttt{mu} and \texttt{g\_mu}:

It is the amount that the repair removes, in mean units, computed on the tracked values.
The cost report uses these values.
```

### M-C-27: Output: depot, idle and intervention (previous wording)
- From: Output Specification, Rules for the Values
- Kind: remark
- Suggested destination: Overview of the problem (Scheduling and maintenance variables)
- Spec keeps: the definitions of an idle day and of the key `idle`, with the marks `I` and `D`

```latex
\textbf{Depot, idle and intervention.}
A row of zeros $x_{i1k} = \cdots = x_{iMk} = x_{i0k} = 0$ is an idle day of train $i$: no mission and no depot.
On a depot day ($x_{i0k} = 1$) each component is repaired, replaced or in the depot without an intervention; the last case is \texttt{idle}.
Hence $x^m_{i\ell k} + x^r_{i\ell k} + \text{idle}_{i\ell k} = x_{i0k}$.
The key \texttt{idle} is $1$ when component $\ell$ rests in the depot without an intervention; the plot shows it with the mark \texttt{D} (Section~\ref{sec:plotting}).
On an idle day of the train (Section~\ref{sec:assignment}) the key is $0$, and the plot shows the mark \texttt{I}.
```

### M-C-28: Output: cost report (previous wording)
- From: Output Specification, Rules for the Values
- Kind: remark
- Suggested destination: Overview of the problem (Cost function)
- Spec keeps: the citation of Equations `eq:a_daycost` and `eq:a_phases` and the relation of `objective` to the report

```latex
\textbf{Cost report.}
The cost $c_k$ of day $k$, and the phase costs $J_{\mathrm{init}}$ and $J_{\mathrm{op}}$, are those of Equations~\eqref{eq:a_daycost} and~\eqref{eq:a_phases} (Section~\ref{sec:objective}), computed from the arrays of the result.
Where $z_{i\ell k}$ is \texttt{NaN} (cells with $C_{R,i\ell} = 0$), it reads $0$.
The value \texttt{objective} is the value of the model objective.
When the variable $u$ has no slack, it equals, up to the tolerance of the solver, \texttt{J\_total} if \texttt{objective\_mode} is \texttt{total}, \texttt{operating\_objective} if it is \texttt{operating\_average}, and \texttt{evaluation\_objective} if it is \texttt{evaluation\_total}; otherwise it is larger by $C_D$ times that slack.
```

### M-C-29: Output: result-table rows, removed wording
- From: Output Specification, Result Dictionary
- Kind: remark
- Suggested destination: Implementation details
- Spec keeps: the rows, with one-line descriptions

```latex
\typerow{\texttt{status}}{---}{Status of the solve: \texttt{optimal}, \texttt{time\_limit}, \texttt{suboptimal}, \texttt{interrupted}, \texttt{solution\_limit}, \texttt{node\_limit}, \texttt{iteration\_limit}, \texttt{infeasible}, \texttt{inf\_or\_unbounded}, \texttt{unbounded}, or \texttt{gurobi\_status\_}$n$ for any other Gurobi status code $n$}
\typerow{\texttt{objective}}{--- or \texttt{null}}{Objective value of the best solution found; \texttt{null} if there is none}
\typerow{\texttt{mip\_gap}}{--- or \texttt{null}}{Relative MIP gap of the best solution at the end of the solve; \texttt{null} if there is no solution. The input key \texttt{mip\_gap} is the gap that was requested}
\typerow{\texttt{bound}}{--- or \texttt{null}}{Best bound of Gurobi, if it is available}

\typerow{\texttt{x}}{$F \times (M{+}1) \times T$}{Assignment: $x_{ijk} = 1$ if train $i$ flies mission $j \ge 1$ on day $k$, and $x_{i0k} = 1$ if it is in the depot. A train with $x_{ijk} = 0$ for every $j \in \{0,\ldots,M\}$ is idle}

\typerow{\texttt{v}}{$F \times L \times T$}{Tracked variance. \texttt{NaN} in memory and \texttt{null} in a file at cells that track no variance (IG, Ga, RF with Mk, Ho or Ch)}
\typerow{\texttt{z}}{$F \times L \times T$}{Repair-cost variable. \texttt{NaN} in memory and \texttt{null} in a file at cells with $C_{R,i\ell} = 0$, where the variable is not built. On the transitory days of the mode \texttt{operating\_average} the value is computed after the solve (below)}
\typerow{\texttt{R}, \texttt{K}}{$F \times L \times T$}{Tracked Hoeffding budget $R$ and Chernoff quantity $K$. \texttt{NaN} in memory and \texttt{null} in a file at the cells that do not track the quantity: $R$ is tracked at RF (Ho) cells, $K$ at RF (Ch) cells}
\typerow{\texttt{g\_mu}, \texttt{g\_v}, \texttt{g\_R}, \texttt{g\_K}}{$F \times L \times T$}{Tracked anchors $g^{\mu}_{i\ell k}$, $g^{v}_{i\ell k}$, $g^{R}_{i\ell k}$ and $g^{K}_{i\ell k}$: the value of the state after the last intervention up to day $k$. \texttt{NaN} in memory and \texttt{null} in a file at the cells that do not track the anchor: only cells with \texttt{repair\_model} \texttt{ard1} track one, \texttt{g\_mu} for every model, \texttt{g\_v} at G and RF (Ca, Be) cells, \texttt{g\_R} at RF (Ho) cells, \texttt{g\_K} at RF (Ch) cells}
```

### M-C-30: Result files: reason of the early check
- From: Output Specification, Result Files
- Kind: rationale
- Suggested destination: Implementation details
- Spec keeps: the rule that the extension and the directory are checked before the solve

```latex
  \item The extension, and the directory of the path, are checked before the model is solved, so that a long solve is not lost. The directory must exist and be writable.
```

### M-C-31: Cell-type notes and YAML example: description and worked remarks
- From: Input File Schema, Keys by Cell Type and YAML Example (`sec:schema_fields`, `sec:schema_example`)
- Kind: example
- Suggested destination: Implementation details
- Spec keeps: one line: the six cells and that the example passes every check; the notes on starting anchors and tangent_ref in the intro of the cell-type table

```latex
Every rainflow cell also needs the key \texttt{bound\_method} when its bound is not Cantelli.
\texttt{anchor\_mu\_0} and the other starting anchors are read only at cells with \texttt{repair\_model} equal to \texttt{ard1}.

The example has $F = 3$ trains, $M = 2$ missions and $L = 2$ components.
The six cells use Gaussian, inverse Gaussian, Wiener, Gamma, rainflow with Bernstein, and rainflow with Chernoff.
It uses an array, a list and a scalar for cell values, a profile of shape $(L, M)$ and a transitory profile of shape $(L, M, H_1)$.
It passes every check of Section~\ref{sec:consistency_checks}.

In the example, the inverse Gaussian cell uses the law \texttt{ard1} with the starting anchor $0.06 \le \mu_0 = 0.10$. The Gamma cell has the starting anchor $0.05 \le \mu_0 = 0.08$ and no \texttt{gamma\_beta\_0}, so that its initial damage is known. The Chernoff cell has $K_0 = 5 \cdot 0.14 = 0.70$ by default, so that its starting anchor $0.50$ is allowed.
The Bernstein cell has the support constant $b = \max(0.05, 0.03, 0.05) = 0.05$, so that $2\kappa b/3 = 0.077 < \tau = 1$ with $\kappa = \ln 10$.
```
