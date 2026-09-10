# coding: utf-8

"""
Variables of the TOPO trigger sensitivity study.

These read the ``topo_feat`` field written by :py:func:`hbw.production.topo_trigger.topo_features`,
which only exists for selectors with ``produce_topo_features = True`` (i.e. ``sl1_topo``).

The muon-pT binning is the study's reference binning, deliberately fine around 24 GeV: the
acceptance gain of interest lies *below* the IsoMu24 turn-on, so the comparison of
``sl_topo_or2`` and ``sl_topo_or3`` in this variable must not be smeared by coarse bins there.
"""

from __future__ import annotations

import order as od

from columnflow.columnar_util import EMPTY_FLOAT
from hbw.util import call_once_on_config


#: reference PT_EDGES
TOPO_PT_EDGES = [10, 15, 20, 22, 24, 26, 28, 30, 35, 40, 50, 70, 100, 150]

#: reference ETA_EDGES
TOPO_ETA_EDGES = [-2.4, -2.1, -1.6, -1.2, -0.9, -0.3, 0.3, 0.9, 1.2, 1.6, 2.1, 2.4]


@call_once_on_config()
def add_topo_variables(config: od.Config) -> None:
    config.add_variable(
        name="topo_mu_pt",
        expression="topo_feat.mu_pt",
        null_value=EMPTY_FLOAT,
        binning=TOPO_PT_EDGES,
        unit="GeV",
        x_title=r"leading isolated muon $p_{T}$",
    )
    # Uniform-width companion to topo_mu_pt. TOPO_PT_EDGES is the reference binning and is kept
    # as the reference, but its widths jump 5 -> 2 -> 5 GeV around the turn-on, and since the
    # y-axis is *entries* (not entries/GeV) that produces a sawtooth at 15--30 GeV which reads
    # as structure in the spectrum and is not. 2 GeV throughout, over the turn-on region.
    config.add_variable(
        name="topo_mu_pt_uni",
        expression="topo_feat.mu_pt",
        null_value=EMPTY_FLOAT,
        binning=(20, 10.0, 50.0),
        unit="GeV",
        x_title=r"leading isolated muon $p_{T}$",
    )
    config.add_variable(
        name="topo_mu_eta",
        expression="topo_feat.mu_eta",
        null_value=EMPTY_FLOAT,
        binning=TOPO_ETA_EDGES,
        x_title=r"leading isolated muon $\eta$",
    )
    config.add_variable(
        name="topo_mu_iso",
        expression="topo_feat.mu_iso",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 0.15),
        x_title=r"leading isolated muon pfRelIso03",
    )
    config.add_variable(
        name="topo_ht",
        expression="topo_feat.ht",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1200.0),
        unit="GeV",
        # NOTE: this is the OFFLINE, muon-cross-cleaned HT over jets with pt > 25 and
        #       |eta| < 2.5 and no jet ID -- not hbw's "ht", and not the online PF HT the
        #       PFHT150 leg cuts on. That difference is real and is studied separately.
        x_title=r"offline $H_{T}$ (topo definition)",
    )
    config.add_variable(
        name="topo_lead_btag",
        expression="topo_feat.lead_btag",
        null_value=EMPTY_FLOAT,
        binning=(25, 0.0, 1.0),
        x_title="leading btagPNetB",
    )
    config.add_variable(
        name="topo_lead_bpt",
        expression="topo_feat.lead_bpt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"leading b-tagged jet $p_{T}$",
    )
    config.add_variable(
        name="topo_n_btag_pnet",
        expression="topo_feat.n_btag_pnet",
        binning=(7, -0.5, 6.5),
        x_title="number of jets with btagPNetB > 0.2",
        discrete_x=True,
    )
    config.add_variable(
        name="topo_l1ht_proxy",
        expression="topo_feat.l1ht_proxy",
        binning=[0, 120, 150, 200, 255, 280, 320, 360, 400],
        unit="GeV",
        x_title=r"highest fired L1 $H_{T}$ threshold",
    )
    config.add_variable(
        name="topo_valid",
        expression="topo_feat.valid",
        binning=(2, -0.5, 1.5),
        x_title="inside the estimator's training support",
        discrete_x=True,
    )
    config.add_variable(
        name="topo_or2",
        expression="topo_or2",
        binning=(2, -0.5, 1.5),
        x_title="OR2 fired (IsoMu24 or Mu12_PFHT150_PNetB)",
        discrete_x=True,
    )
