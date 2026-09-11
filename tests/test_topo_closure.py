# coding: utf-8

"""
unittests for the TOPO emulation closure test

These check the two assumptions the closure arithmetic rests on. Both are invisible in normal use --
a broken one produces plausible numbers rather than an error -- so they are worth pinning.
"""

import unittest

from columnflow.util import maybe_import
from columnflow.hist_util import create_hist_from_variables

from hbw.weight.topo_closure import (
    TOPO_CLOSURE_ARMS, TOPO_CLOSURE_TARGETS, TOPO_PREDICTION_ONLY,
)

import order as od

np = maybe_import("numpy")
hist = maybe_import("hist")


class TopoClosureTest(unittest.TestCase):

    def test_arm_names_are_unique_and_complete(self):
        """The arm axis is addressed by index on fill and by name on read; a duplicate would make
        the two disagree silently."""
        self.assertEqual(len(TOPO_CLOSURE_ARMS), len(set(TOPO_CLOSURE_ARMS)))
        for est in TOPO_CLOSURE_TARGETS:
            for prefix in ("dir", "emu", "dif", "std"):
                self.assertIn(f"{prefix}_{est}", TOPO_CLOSURE_ARMS)
        for est in TOPO_PREDICTION_ONLY:
            self.assertIn(f"emu_{est}", TOPO_CLOSURE_ARMS)
            # a prediction-only estimator must NOT get a residual arm: there is no truth to
            # subtract, and an arm named dif_* would invite exactly that mistake
            self.assertNotIn(f"dif_{est}", TOPO_CLOSURE_ARMS)
        self.assertIn("den", TOPO_CLOSURE_ARMS)
        self.assertIn("den_all", TOPO_CLOSURE_ARMS)

    def test_paired_error_identity(self):
        """
        ``hist``'s Weight storage must accumulate sum(w) in values() and sum(w**2) in variances().

        The whole paired-error treatment depends on it: filling with the per-event difference
        ``d = f - y`` then gives ``var(d) = E[d^2] - E[d]^2`` for free, and the residual's error is
        ``sqrt(var(d) / N)``. If a future ``hist`` changed this, the closure would keep producing
        numbers -- wrong ones, and too small.
        """
        rng = np.random.default_rng(1234)
        n = 5000
        x = rng.uniform(0.0, 10.0, n)
        y = (rng.uniform(size=n) < 0.7).astype(float)      # stored decision
        f = np.clip(y + rng.normal(0.0, 0.2, n), 0.0, 1.0)  # emulated efficiency
        d = f - y

        var_inst = od.Variable(name="x", expression="x", binning=(1, 0.0, 10.0))
        h = create_hist_from_variables(var_inst, weight=True)
        h.fill(x=x, weight=d)

        self.assertAlmostEqual(float(h.values(flow=True).sum()), float(d.sum()), places=6)
        self.assertAlmostEqual(float(h.variances(flow=True).sum()), float((d ** 2).sum()), places=6)

        delta = float(h.values(flow=True).sum()) / n
        var_d = float(h.variances(flow=True).sum()) / n - delta ** 2
        sigma = np.sqrt(var_d / n)

        self.assertAlmostEqual(delta, float(d.mean()), places=9)
        # this is the number the task quotes; it must be the paired error, not a quadrature sum
        self.assertAlmostEqual(sigma, float(d.std() / np.sqrt(n)), places=9)

        naive = np.sqrt(f.std() ** 2 + y.std() ** 2) / np.sqrt(n)
        self.assertLess(sigma, naive)

    def test_flow_is_not_optional(self):
        """
        EMPTY_FLOAT is finite, so undefined entries land in underflow instead of being dropped.

        An integral taken without ``flow=True`` therefore loses them, which biases a per-sample
        efficiency without raising anything. This pins the difference so the convention cannot be
        relaxed by accident.
        """
        var_inst = od.Variable(name="x", expression="x", binning=(2, 0.0, 10.0))
        h = create_hist_from_variables(var_inst, weight=True)
        h.fill(x=np.array([1.0, 5.0, -99999.0]), weight=np.ones(3))

        self.assertEqual(float(h.values(flow=True).sum()), 3.0)
        self.assertEqual(float(h.values(flow=False).sum()), 2.0)


if __name__ == "__main__":
    unittest.main()
