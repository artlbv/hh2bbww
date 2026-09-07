# coding: utf-8

"""
Derived selectors for the TOPO trigger sensitivity study (SL channel, MC only).

Three things distinguish these from stock ``sl1``:

* **No HLT requirement.** ``trigger={"e": [], "mu": []}`` makes ``sl_lepton_selection`` fall back to
  an all-True mask (see :py:mod:`hbw.selection.sl_remastered`), so the trigger enters only as an
  event weight downstream. Without this the emulated weight would be applied on top of an
  already-triggered sample and the OR3-vs-OR2 comparison would be meaningless.
* **A muon pT threshold of 10 GeV**, which is the entire point of the study: the TOPO acceptance
  gain lives *below* the IsoMu24 turn-on. Stock ``mu_pt = 25`` would hide it. 10 GeV is also the
  supported floor of the muon scale factors (``min_pt=10.0`` in the config's muon SF names), so
  going lower would break the SF chain.
* **The offline TOPO features**, produced inside ``cf.SelectEvents`` because they need the raw jet
  collection -- see :py:mod:`hbw.production.topo_trigger`.

Note that restricting to the muon channel is a *category* cut, not a selection cut: ``catid_1mu``
and the ``1mu`` category already exist, so the electron channel is kept as a free null test (TOPO
has a muon leg, so ``sr__1e`` must show no OR3 gain).

``n_btag`` is deliberately left at hbw's default rather than raised to 2 to reproduce the note's
">= 2 b-tags" phase space: hbw's ``n_btag`` counts ``btagUParTAK4B`` at the medium working point for
``year >= 2024``, while the note counts ``btagPNetB > 0.2``. The note's counting is reproduced by
``topo_feat.n_btag_pnet`` instead, which is what ``sl1_topo_pps`` cuts on downstream.
"""

from hbw.selection.sl_remastered import sl1


sl1_topo = sl1.derive("sl1_topo", cls_dict={
    # the paper's MU_PT, and the floor of the muon SF support
    "mu_pt": 10.,
    "mu2_pt": 10.,
    "ele_pt": 31.,
    "ele2_pt": 15.,
    # NO HLT cut -- the trigger enters only as a weight
    "trigger": {"e": [], "mu": []},
    "n_jet": 3,
    "n_btag": 1,
    "produce_topo_features": True,
    "version": 0,
})
