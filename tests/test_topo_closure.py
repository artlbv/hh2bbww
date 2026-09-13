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
from hbw.production.topo_trigger import (
    TOPO_ESTIMATORS, TOPO_RESIDUAL_ESTIMATOR, TOPO_WEIGHT_CHOICES, topo_or3_weights,
)
from hbw.tasks.topo_closure import DIFFERENTIAL_FLOW_OFFSET

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
        # both gains, against the stored and against the emulated OR2
        self.assertIn("gain_or3", TOPO_CLOSURE_ARMS)
        self.assertIn("gain_emu", TOPO_CLOSURE_ARMS)

    def test_residual_cannot_become_the_weight(self):
        """
        ``topo_res`` targets ``TOPO & ~OR2``, a joint probability and not an efficiency, so it is
        meaningless as a weight on its own -- it needs the ``1 / (1 - eps_OR2)`` conditioning and
        the stored OR2 term around it. The applied weight is a single directly fitted efficiency
        instead, so that no scale factor for the online-b-tagged ``Mu12`` leg rides on a term worth
        most of the weight. Pin both halves: the residual is not offerable as a weight, and every
        choice that is offered exists in the bundle.
        """
        self.assertNotIn(TOPO_RESIDUAL_ESTIMATOR, TOPO_WEIGHT_CHOICES)
        for name in TOPO_WEIGHT_CHOICES:
            self.assertIn(name, TOPO_ESTIMATORS)
        self.assertIn(topo_or3_weights.weight_estimator, TOPO_WEIGHT_CHOICES)

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

    def test_differential_flow_offset(self):
        """
        Every per-bin array the task publishes is ``values(flow=True)``, so it is one entry longer
        than ``bin_edges`` has bins and index 0 is the underflow.

        On an integer-binned axis this reads as ``k -> index k + 1``. Getting it wrong does not
        raise and does not look wrong: the columns simply shift, and on a b-tag multiplicity a
        shifted table is indistinguishable from a table taken at a different b-tag cut. The check
        that catches it is an independently known quantity, so this pins the offset against the
        exact axis the b-tag breakdown is read off.
        """
        # the binning of topo_n_btag_pnet: integers 0..6 on half-integer edges
        n_btag_bins = 7
        var_inst = od.Variable(
            name="topo_n_btag_pnet",
            expression="topo_n_btag_pnet",
            binning=(n_btag_bins, -0.5, n_btag_bins - 0.5),
        )
        h = create_hist_from_variables(var_inst, weight=True)

        # a known multiplicity spectrum, plus one undefined entry that must land in underflow
        counts = {0: 3.0, 1: 11.0, 2: 17.0, 3: 5.0}
        x = np.concatenate([np.full(int(c), float(k)) for k, c in counts.items()] +
                           [np.array([-99999.0])])
        h.fill(topo_n_btag_pnet=x, weight=np.ones(len(x)))

        values = h.values(flow=True)
        edges = h.axes[0].edges

        # the layout claim itself: one underflow plus one overflow around len(edges) - 1 real bins
        self.assertEqual(len(values), len(edges) - 1 + 2)
        self.assertEqual(DIFFERENTIAL_FLOW_OFFSET, 1)
        self.assertEqual(float(values[0]), 1.0)  # the EMPTY_FLOAT entry, not a zero-b-tag event

        for k, c in counts.items():
            self.assertEqual(float(values[k + DIFFERENTIAL_FLOW_OFFSET]), c)

        # and the aggregate the b-tag breakdown actually quotes
        n_total = sum(counts.values())
        p_ge2 = sum(float(values[k + DIFFERENTIAL_FLOW_OFFSET])
                    for k in range(2, n_btag_bins)) / n_total
        self.assertAlmostEqual(p_ge2, (counts[2] + counts[3]) / n_total, places=12)

        # the failure this guards against: read without the offset, 0b picks up the sentinel and
        # every column slides by one, which is a plausible-looking table and not an error
        self.assertNotEqual(float(values[0]), counts[0])


if __name__ == "__main__":
    unittest.main()
