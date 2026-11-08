"""Eq. c_phase exists twice: config._lookup (vectorised, WP4) and base.phase_day (WP5).

Usage: python phase_agreement.py
For every H1, H2 in 1..7, with and without a transitory profile, both must give the
same value on every day 1..T. The profile values encode (phase, day index), so a
wrong index or a wrong phase gives a different number. Prints the number of cases
and of disagreements; the last line is ``disagreements: 0`` when both agree.
"""
import pathlib
import sys

import numpy as np

from load_base import load

WORKTREE_SRC = pathlib.Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, str(WORKTREE_SRC))
from fleet_management.config import _lookup  # noqa: E402

base = load()
cases = disagreements = 0
F, L, M = 2, 2, 3
for H1 in range(1, 8):
    for H2 in range(1, 8):
        T = H1 + H2
        # value = 1000*train + 100*component + 10*mission + day index on the profile axis
        idx = np.indices((F, L, M))
        stem = 1000 * idx[0] + 100 * idx[1] + 10 * idx[2]
        op = stem[..., None] + np.arange(H2)[None, None, None, :] + 0.0
        tr = -(stem[..., None] + np.arange(H1)[None, None, None, :] + 1.0)  # negative: transitory
        for trans in (None, tr):
            cases += 1
            vectorised = _lookup(op, trans, H1, H2)
            if vectorised.shape != (F, L, M, T):
                disagreements += 1
                print(f"shape H1={H1} H2={H2}: {vectorised.shape}")
                continue
            for k in range(T):
                use_trans, h = base.phase_day(k, H1, H2, trans is not None)
                source = trans if use_trans else op
                if not np.array_equal(vectorised[..., k], source[..., h]):
                    disagreements += 1
                    print(f"H1={H1} H2={H2} trans={trans is not None} day index {k}")
print(f"cases: {cases}")
print(f"disagreements: {disagreements}")
