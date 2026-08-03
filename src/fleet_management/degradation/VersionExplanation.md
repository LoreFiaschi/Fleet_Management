This subfolder holds the new implementation of Gamma after the Midterm on 28.07.2026
src/fleet_management still has dashboard_gamma.py, degradation_models.py, gamma_process.py, model_registry.py which are older and not supported at the moment.

Documentation of Gamma implementation 03.08.2026

D_ilk ~ Gamma(A_ilk, beta_l)   with beta_l = const
Idle            A_k = A_k-1
Mission j       A_k = A_k-1 + beta * mu^(j)
Replacement     A_k = beta * mu^(new)
Imperfect repairNot supported yet

Pr(D_k > tau) <= epsilon
A_2H <= A_H
Mission increments are non-negative and independent
All accumulated increments for one component share the same beta

