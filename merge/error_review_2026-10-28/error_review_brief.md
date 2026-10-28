# Brief: find the errors in one block of the spec and of the scoping document

## Context

- Project: the Python package `fleet_management` schedules a fleet whose components degrade stochastically;
  the schedule is a mixed-integer program solved with Gurobi. Repository:
  `C:\Users\utente\Documents\GitHub\Fleet_Management`.
- Two documents describe it, and each piece of information lives in exactly one of them:
  - `spec/spec.tex` (PDF: `merge/spec_drafts/build_final/spec.pdf`) is a Software Design Document: inputs,
    checks, variables, rows, constants, outputs, tools. **It is assembled** from the fragments
    `merge/spec_drafts/fragments_A.tex` (overview, notation, variables, objective, general constraints, linear
    forms, repair cost, damage penalty, loop), `fragments_B.tex` (the five models), `fragments_C.tex` (inputs,
    output, schema), `fragments_D.tex` (architecture, tools) and `claude_full_problem.tex` (full problem).
    A fix to the spec must be written against the fragment, not against `spec/spec.tex`.
  - `scoping.tex` (PDF: `scoping.pdf`) is the research document: aim, background, literature, derivations
    and proofs (in the appendix "Proofs and derivations"), and the reasons for each choice.
- The owner is a mathematician and wants a rigorous, picky check: explicit constants, every step valid.
  The owner's first language is not English.

## The task

Check **your block** (below) for **errors**, and propose a fix for each. Do not edit any repository file:
the orchestrator verifies your findings and applies the fixes.

An error is one of these:
1. **Mathematics:** a wrong formula, sign, factor, index, inequality direction, constant or condition; a
   proof or derivation step that does not follow; a hypothesis that is missing or not used where claimed; a
   claim that is false (give a counterexample).
2. **Numbers:** a numeric value or worked example that does not reproduce. Recompute every number of your
   block that can be recomputed, with a script.
3. **Consistency:** the two documents, or two places of one document, define or state the same thing
   differently (a formula, a default, a range, a name, a key, a condition, a table entry, a count).
   Check your spec part against your scoping part, and both against the places they cite.
4. **References:** a `\ref`, `\eqref` or "Section/Table/Equation X" that points to the wrong place, or a
   statement that cites a result that does not give it.
5. **Language that changes the meaning:** a sentence that says something else than intended, an undefined
   symbol or term, a term used for two concepts. Not style.

Do **not** report: style or wording preferences, structure suggestions, missing features, design choices
you would make differently, or claims about cited literature (the sources are not available; the only one
in `literature/` is Hoeffding 1963). No minor comment added only to look thorough.

## How to work

1. First action: write the skeleton of your findings file (path below) with a `Status: in progress` line.
2. Read your block in full, in the `.tex` source, and render pages of the PDF when the layout matters
   (`pdftoppm -png -r 110 -f N -l N file.pdf out`, then read the PNG).
3. For each candidate error, verify it before you write it down: derive it by hand, and recompute numbers
   with Python. Use `python` (not `python -I`): numpy, scipy, mpmath and sympy are installed for it.
   Save each check script in the scratchpad (below) and name it in the finding.
4. Append each finding to the file as soon as it is verified (one unit at a time; at most about 150 lines
   per edit). Update the status line as you go.
5. When something outside your block is needed to judge (a definition, a cited equation), read it there.

## Findings file format

```
Status: done (N findings) | in progress (last checked: ...)

### F1 [error] Short title
- Where: scoping.tex:1234 (Proposition prop:xxx) and spec/spec.tex:567 (fragments_B.tex:89)
- Problem: one or two sentences.
- Evidence: the derivation, or the script (path) and its output.
- Fix: file `merge/spec_drafts/fragments_B.tex` (or `scoping.tex`)
  - old: `exact text, unique in the file, copied verbatim`
  - new: `replacement text`
- Confidence: high | medium
```
Severity tags: `[error]` (mathematics, numbers, wrong reference), `[inconsistency]` (two places disagree;
say which one is right and why), `[meaning]` (language that changes the meaning).
Order the findings by severity, most severe first, when you finish. If you are not sure that something is
an error, put it under a final heading `## Doubts` with the reason, and no fix.

## Writing rules

Short sentences, common words, active voice; one term for one concept; no idioms. The fixes follow the same
rules and the style of the surrounding text. A fix changes only what the error needs.

## Constraints

- Do not spawn sub-agents.
- Do not delete any file; write intermediate outputs only to the scratchpad
  `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\`.
- Do not edit any file in the repository.
- Never put the user's name or email address in any request: not in headers, URLs or payloads.
  (This task needs no network access.)
- Final chat reply: at most 20 lines (counts per severity and the three most serious findings); the content
  goes into the findings file.
