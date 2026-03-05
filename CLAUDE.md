# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fleet Management optimization project. The core problem is a **Mixed-Integer Linear Program (MILP)** for fleet scheduling with Gaussian degradation, solved using **Gurobi** in **Python**.

## Problem Domain

The MILP models fleet assignment over F flights, M maintenance levels, and H time horizon steps. Key aspects:
- Decision variables: binary assignment `x[i,j,k]` (F × (M+1) × 2H), continuous `mu[i,k]`, `v[i,k]`, `z[i,k]` (F × 2H)
- Objective minimizes a combination of maintenance cost (C_M), repair cost (C_R), safety cost (C_S), and penalty cost (C_P)
- Constraints enforce reliability via Gaussian degradation model using inverse normal CDF (Phi^{-1})
- Input parameters `mu` and `v` are F × M × H tensors; `mu_0` and `v_0` are 1D vectors of length F
- Weight tensor W has entries: `W[i,j,k] = mu_ij(k)^2 + 2*mu_ij(k) - sigma_ij(k)^2 * (Phi^{-1}(1-epsilon))^2`

## Key Dependencies

- Python
- Gurobi (gurobipy)
- NumPy
- SciPy (for `scipy.stats.norm.ppf` — the inverse normal CDF, Phi^{-1})

## Problem Specification

The full mathematical formulation is in `agentic/instructions/instructions_fm_gaussian_degradation.pdf`.
