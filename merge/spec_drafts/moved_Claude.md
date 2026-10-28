# Blocks moved out of the spec by Claude (26 Oct 2026)

### M-Cl-1: Open questions
- From: section Open Questions (sec:open), Claude's part of the spec.
- Kind: open question.
- Suggested destination: Implementation details, or a short subsection on open questions near it.
- Spec keeps: nothing (the section leaves the spec).

```latex
This section lists the design questions that this specification leaves open.
The planned work of the project is listed in its to-do file,
\texttt{Things to change.txt}.

\begin{itemize}[leftmargin=1.5em]
  \item \textbf{Hierarchical failure-probability bounds.}
        Every reliability row bounds the failure probability of one component:
        $\Pr(D_{i\ell k} > \tau_{i\ell}) \leq \varepsilon_{i\ell}$ for all $i, \ell, k$.
        A user may want to state the requirement at a coarser level:
        \begin{itemize}
          \item \textbf{Train level:} $\Pr(\exists\,\ell : D_{i\ell k} > \tau_{i\ell}) \leq \varepsilon_i$
                for all $i, k$, the probability that \emph{any} component of train $i$
                fails. Under independence of the components this reads
                $1 - \prod_\ell (1 - \Pr(D_{i\ell k} > \tau_{i\ell})) \leq \varepsilon_i$,
                a nonlinear row that couples the components of a train.
          \item \textbf{Fleet level:} $\Pr(\exists\,i,\ell : D_{i\ell k} > \tau_{i\ell}) \leq \varepsilon$
                for all $k$, the probability that \emph{any} component of the fleet fails.
                Under independence this row couples all trains.
        \end{itemize}
        Both require either union-bound approximations (conservative and linear) or
        product-form rows (exact and nonlinear). The per-component levels give both
        through the union bound: $\varepsilon_{i\ell} \leq \varepsilon_i / L$ for every
        $\ell$ gives the train level, and $\varepsilon_{i\ell} \leq \varepsilon/(FL)$
        for every cell gives the fleet level.
  \item \textbf{Sampling error of the empirical inputs.}
        Rainflow counting yields \emph{estimates} of the per-mission mean, variance,
        support width and cumulant generating function, not exact values. A guaranteed
        reliability bound needs one-sided confidence margins on these inputs.
  \item \textbf{Gamma cap over the reachable states.}
        The cap of Section~\ref{sec:dyn_gamma} is the largest cap that is safe for every
        member of the class of true states defined there. The states that a schedule can
        reach form a smaller set, because the tracked mean also satisfies the cap on every
        earlier day. For a cell with one rate and $\beta^*\tau$ below the threshold
        $t_1(\varepsilon)$ of Section~\ref{sec:dyn_gamma}, a larger safe cap may exist.
        Its value is not known.
\end{itemize}
```

### M-Cl-2: References of the spec
- From: the reference list of the spec (four sources: Doyen and Gaudoin 2004, Mercier and Castro 2019,
  Roosta-Khorasani and Szekely 2015, Szekely and Bakirov 2003).
- Kind: literature.
- Suggested destination: `scoping_bib.bib` (already holds all four; nothing to add).
- Spec keeps: nothing.
