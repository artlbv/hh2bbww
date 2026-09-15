# coding: utf-8

"""
Muon identification variables of the TOPO non-prompt trigger study.

These read per-muon ``Muon`` fields that are only kept from the ``topo_feat2`` reduction onward
(``config_run2.py`` keep_columns: the promptMVA inputs and ``Muon.pnScore_{prompt,heavy,light,tau}``).
Plotting them at an earlier version silently yields empty histograms, because the columns are
simply absent from the reduced files rather than present-and-null.

WHY ``[:, 0]`` IS UNAMBIGUOUS HERE. After the ``sl1_topo`` reduction the ``Muon`` collection holds
AT MOST ONE muon per event (measured on tt_fh: mean 0.418, max 1), so the leading muon is the
only muon, and ``Muon.<f>[:, 0]`` agrees with ``topo_feat.mu_pt`` on 99.93% of events that have
one. The residual 0.07% are events whose single muon fails the study's isolation/kinematic cut,
where ``topo_feat`` carries EMPTY_FLOAT instead; those land in the underflow via *null_value*.

WHY THESE VARIABLES EXIST AT ALL. The emulation-closure failure of the shipped estimator is a
function of muon ORIGIN, not of muon kinematics: it closes on prompt muons and misses by ~+24 pp
on muons from b decays. The estimator is fitted on mu_pt/mu_eta/mu_iso (plus event-level terms
for the non-isomu24 legs), none of which separate those populations. These are the variables
that do, so they are the ones worth looking at differentially.
"""

from __future__ import annotations

import order as od

from columnflow.columnar_util import EMPTY_FLOAT
from hbw.util import call_once_on_config


@call_once_on_config()
def add_topo_muid_variables(config: od.Config) -> None:
    # The four ParticleNet lepton scores. Each is a probability in [0, 1] and is strongly peaked
    # at both ends, so a uniform 40-bin axis is readable while a coarser one would merge the
    # peak at 1 -- which for the heavy score is exactly the population the study is about -- into
    # its neighbour.
    for short, title in [
        ("prompt", "prompt"),
        ("heavy", "heavy flavour"),
        ("light", "light flavour"),
        ("tau", r"$\tau$"),
    ]:
        config.add_variable(
            name=f"muon0_pnscore_{short}",
            expression=f"Muon.pnScore_{short}[:, 0]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0.0, 1.0),
            x_title=f"muon PNet {title} score",
        )

    config.add_variable(
        name="muon0_promptmva",
        expression="Muon.promptMVA[:, 0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1.0),
        x_title="muon promptMVA",
    )

    # jetRelIso = pt_mu / pt_jet - 1 against the muon's host jet. The value -1 is a SENTINEL
    # meaning "no host jet", not a measurement, and it is a point rather than a region -- the
    # band (-1, -1/3) is physical, where the host jet's pt falls below the muon's after lepton
    # subtraction and JEC. The axis therefore starts just below -1 with a bin edge at -0.95, so
    # the sentinel occupies its own first bin and does not contaminate the physical band next to
    # it. An earlier script in this study cut jetRelIso >= -0.5 intending "has a host jet" and
    # discarded real host jets as a result; the binning is chosen so that the plot cannot repeat
    # that mistake visually.
    config.add_variable(
        name="muon0_jetreliso",
        expression="Muon.jetRelIso[:, 0]",
        null_value=EMPTY_FLOAT,
        binning=[-1.05, -0.95] + [round(-0.95 + 0.1 * i, 2) for i in range(1, 41)],
        x_title="muon jetRelIso",
    )
