# Blocks moved out of the spec by writer D (26 Oct 2026)

Source of the blocks: `fragments_D.tex` before the split of specification and scoping (copy in the session scratchpad). The text of each block is verbatim.

### M-D-1: Why a ranking by the cost per day compares horizons
- From: Horizon Sweeps, Cost Measures and the Ranking Criteria (sec:sweep_criteria).
- Kind: rationale.
- Suggested destination: Implementation details (horizon sweeps), or Overview of the problem (Cost function).
- Spec keeps: The rule of the ranking (which value each function ranks by).

```latex
The loop constraint (Section~\ref{sec:loop}) makes the operating phase repeat,
so $\bar J_{\mathrm{op}}$ is the cost per day of the repeating regime.
It is comparable across values of $H_2$, whereas a raw objective over $T$ days is not.

With an evaluation horizon it ranks by $P_E$, which also prices the transitory phase,
so it is the criterion that compares different values of $H_1$.
```

### M-D-2: Meaning of the projected cost and of the operating-average objective
- From: Horizon Sweeps, Cost Measures and the Ranking Criteria (sec:sweep_criteria).
- Kind: rationale.
- Suggested destination: Overview of the problem (Cost function).
- Spec keeps: The formulas of the two objectives and of $P_E$.

```latex
The transitory phase has no cost in this objective, but it is still constrained.

$P_E$ is the cost of the transitory phase, then of the operating phase repeated until day $E$.
```

### M-D-3: What the proof of a case certifies
- From: Horizon Sweeps, Cost Measures and the Ranking Criteria (sec:sweep_criteria).
- Kind: remark.
- Suggested destination: Implementation details (horizon sweeps).
- Spec keeps: The proof rule (proven means status optimal at the MIP gap used for the case).

```latex
A proven case is optimal within that gap, not exactly optimal.
The sweep report records the requested gap of each case, so a reader sees how strong the proof is.

\emph{Remark (what the proof certifies).}
The status \texttt{"optimal"} certifies the objective that the solve minimised.
This objective is $\bar J_{\mathrm{op}} + C_D\,u$ in the mode \texttt{operating\_average} and $P_E + C_D\,u$ in the mode
\texttt{evaluation\_total}, while the ranking value is $\bar J_{\mathrm{op}}$ or $P_E$ alone.
The sweep therefore ranks by cost and uses the damage penalty only inside each solve.
```

### M-D-4: Consequences of the gap threshold of the early stop
- From: Horizon Sweeps, Early Stop (sec:sweep_stop).
- Kind: remark.
- Suggested destination: Implementation details (horizon sweeps).
- Spec keeps: The definition of a qualifying case and of the threshold.

```latex
With the default, a case qualifies when Gurobi has met the gap that was requested for it.

A case that does not qualify, for example one that stopped at the time limit above its requested gap, is recorded,
but it cannot trigger a stop.
```

### M-D-5: Why parallel cases are refused with warm starts and with the early stop
- From: Horizon Sweeps, Parallel Cases (sec:sweep_parallel).
- Kind: rationale.
- Suggested destination: Implementation details (horizon sweeps).
- Spec keeps: The refusal rule, and: each running solve uses one Gurobi licence token.

```latex
Each running solve uses one Gurobi licence token, so a licence with one token allows only $\texttt{n\_workers} = 1$.
The cases do not use each other's results, so the option is refused together with warm starts between cases,
and together with the early stop, which needs the preceding cases before it starts the following ones.
```

### M-D-6: A warm start changes the time of a solve, not the model
- From: Warm Starts, introduction (sec:warm_starts).
- Kind: remark.
- Suggested destination: Implementation details (warm starts).
- Spec keeps: nothing

```latex
A warm start changes the time of a solve, not the model: it does not change the feasible set or the objective.
```

### M-D-7: Properties of the day mapping of a warm start
- From: Warm Starts, Day Mapping (sec:warm_map).
- Kind: rationale.
- Suggested destination: Implementation details (warm starts).
- Spec keeps: The formula $\sigma(k)$ and the mapped values of each variable.

```latex
So the transitory days keep their index, and the operating days repeat the operating phase of the source cyclically.
A shorter operating phase can seed a longer one.
When $H_1 > H_1'$, the days $H_1'+1,\ldots,H_1$ all take the first operating day of the source.
When the source has the same sizes $F$, $M$, $L$, the mapped schedule copies whole days of the source,
so it keeps the one-activity rows, the mission coverage and the depot-day rows of every day.
It need not satisfy the dynamics or the loop rows of the horizons of the problem.
```

### M-D-8: Why hints help and what Gurobi does with a start
- From: Warm Starts, Start, Hints and Completion (sec:warm_start_modes).
- Kind: rationale.
- Suggested destination: Implementation details (warm starts).
- Spec keeps: Which attribute each mode sets, and that a hint needs no feasible point.

```latex
The continuous variables get no value. Gurobi tries to complete the start by itself, with limited effort,
and discards a start that it cannot make feasible.

Gurobi uses hints to guide its heuristics and its branching during the whole solve.
A hint needs no feasible point, so it also helps when the mapped schedule does not fit the horizons of the problem
and Gurobi discards it as a start. The hint priority \texttt{VarHintPri} is not set.
```

### M-D-9: Status of the relaxed solution
- From: Warm Starts, Relaxation Start (sec:warm_relaxation).
- Kind: remark.
- Suggested destination: Implementation details (warm starts).
- Spec keeps: The relaxed solution gives integer values only; it is not a feasible point of the original model.

```latex
The relaxed solution is not a feasible point of the original model.
It gives integer values only, and the original model must find its own first solution.
```

### M-D-10: Aim of the greedy heuristic and independence of the cells
- From: Warm Starts, Greedy Heuristic (sec:greedy).
- Kind: rationale.
- Suggested destination: Implementation details (warm starts).
- Spec keeps: The construction pass, the loop pass and the report.

```latex
It builds a schedule that tries to satisfy the rows of the model. It does not try to give a low cost.

After the construction pass the missions are fixed, so the cells are independent except through the depot days of a train.
```

### M-D-11: Relation of the cluster scripts to the specification
- From: File & Folder Structure, Cluster jobs (sec:files).
- Kind: remark.
- Suggested destination: Implementation details.
- Spec keeps: The list of the two job scripts and the script of examples/regression/ that each runs.

```latex
Both scripts call only the public API of Section~\ref{sec:api}. The behaviour that this specification defines does not depend on them.
```

### M-D-12: One route for every fleet
- From: File & Folder Structure, Public API (sec:api).
- Kind: comparison.
- Suggested destination: Implementation details.
- Spec keeps: solve() has one route for every model (stated in the steps).

```latex
Every fleet, whatever its models, goes through the same route: no model has a separate solver.
```

### M-D-13: Why the tracked means are plotted
- From: File & Folder Structure, Plot Layout (sec:plotting).
- Kind: rationale.
- Suggested destination: Implementation details.
- Spec keeps: The colour scale and the colour-bar label.

```latex
    A colour bar is labelled ``tracked mean / failure threshold''. The values are the tracked means, which are upper bounds of the expected degradation
    (Section~\ref{sec:encoding}).
```

### M-D-14: Why the objective check of validate is one-sided, and why the loop checks use the saved states
- From: Validation Tools, Structural Validator (sec:validation_structural).
- Kind: rationale.
- Suggested destination: Implementation details (validation).
- Spec keeps: The formula of each check and its violation.

```latex
The saved $u$ is the maximum of Eq.~\eqref{eq:c_u}, while the solver objective contains the variable $u$, which can lie above this maximum in a solution that is not optimal.
So the check is one-sided: it tests that the reported objective is not below the cost of the reported solution. It does not test optimality.

The loop checks use the saved tracked states, which satisfy the loop row, and not the reference trajectory:
the reference trajectory of a feasible schedule can violate the loop row (Section~\ref{sec:encoding}).
```

### M-D-15: Reading of the margin in the dashboards
- From: Validation Tools, Dashboards (sec:validation_dashboards).
- Kind: remark.
- Suggested destination: Implementation details (validation).
- Spec keeps: The definition of the margin and its sign rule.

```latex
A margin of at least $-\texttt{tolerance}$ means that the row holds. The exact form is used also when the solve used the tangent or
piecewise-linear form, so the margin shows the true slack.

The margins of cells with different models and bounds are not comparable in size; only the sign is.
```

### M-D-16: The Moschopoulos series and its bounds
- From: Validation Tools, Validators of the Gamma Model, exact validator (sec:validation_gamma).
- Kind: derivation.
- Suggested destination: Stochastic orderings (within each degradation family), or Implementation details.
- Spec keeps: The tail is computed by the Moschopoulos series; the validator uses the partial sum plus the remaining mixture mass as the upper bound.

```latex
It computes the tail $\Pr(D_k > \tau)$ of the sum of the terms with the Moschopoulos series. The series writes the law of a sum of
independent Gamma variables with different rates as a mixture of Gamma laws with non-negative weights. When all rates are equal, the tail is the Gamma tail.
The partial sum of the series is a lower bound of the tail. The validator adds the mixture weight that remains, which is at most \texttt{convolution\_tolerance},
and so obtains a certified upper bound of the tail. If the series does not reach this tolerance within \texttt{max\_series\_terms} terms,

(The saved values may be larger: they are upper bounds.)
```

### M-D-17: Status of the Monte Carlo result
- From: Validation Tools, Validators of the Gamma Model, Monte Carlo validator (sec:validation_gamma).
- Kind: remark.
- Suggested destination: Implementation details (validation).
- Spec keeps: The report is valid when the observed schedule failure rate is at most the target.

```latex
The result is an empirical stress test and not a proof: zero failures in $n$ runs does not show that the true rate is below $1/n$.
```

### M-D-18: Introductory sentences of the sweep, file and plot sections
- From: Horizon Sweeps (sec:sweeps); File & Folder Structure (sec:files); Plotting API and Layout (sec:plotting); Dashboards (sec:validation_dashboards).
- Kind: remark.
- Suggested destination: Implementation details.
- Spec keeps: The names of the functions, modules and files, in the signatures and the layout.

```latex
A single call of \texttt{solve()} uses one horizon pair $H = [H_1, H_2]$ (Section~\ref{sec:notation}).
A \emph{horizon sweep} solves one scenario for several pairs and compares the solved cases.

The package \texttt{fleet\_management} has one module per task. The shared Gurobi model and the registry of cell builders
are in \texttt{base.py}; each degradation model has its own module.

The package has one schedule plot and one sweep plot, both in \texttt{utils/plotter.py}.

The dashboards are Streamlit pages (Streamlit is a Python library for web dashboards; it is in the extra \texttt{dashboard}).
```

### M-D-19: Consequences of rules that the spec states once
- From: Warm Starts in Sweeps (sec:warm_sweeps); Greedy Heuristic (sec:greedy); Report, Checkpoint and Errors (sec:sweep_report); Cell-Builder Interface (sec:builders).
- Kind: remark.
- Suggested destination: Implementation details.
- Spec keeps: The rules themselves: warm start refused with parallel cases; a start with failures is returned; the report is written as YAML; the checkpoint; $\mu$ is defined by every builder.

```latex
Warm starts between cases need the preceding results, so they are refused together with parallel cases (Section~\ref{sec:sweep_parallel}).

A start with failures is still returned. Its completion can then end with the status \texttt{"infeasible"} with an IIS.

The report is a mapping of plain Python types. The file is written with a safe YAML writer, so it
contains no arrays and no Python objects.

If a job ends before the run is complete, the file is a valid report of the cases evaluated so far.

Every builder must define the shared variable $\mu_{i\ell k}$ for all $k = 1,\ldots,T$ of each of its cells, because the damage regularisation
and the output read it. After the optimisation, the shared extraction reads the variables, and then each model's \texttt{extract} hook runs.
```
