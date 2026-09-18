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

    # Muon pt in the exact bins requested for the heavy-score yield split (Artur, 2026-09-18),
    # meant to be used as the FIRST axis of the 2D variable
    # "muon0_pt-muon0_pnscore_heavy". A 2D histogram is required rather than two 1D ones:
    # the yield with heavy < 0.2 INSIDE a pt bin needs the joint distribution, and pt and the
    # heavy score are strongly correlated. Factorising the marginals was measured to be wrong by
    # up to 42 pp (IsoMu24, 12-15 GeV: 41.5% true vs 84.0% inclusive).
    #
    # EDGES. The requested bins are 12-15, 15-20, 20-25, 25+. Two extra edges are added so that
    # no event lands in a flow bin and is silently lost when the histogram is summed:
    #   * 0-10 and 10-12 below, because the muon collection keeps pt >= 5 (hbw/selection/
    #     lepton.py:101) and 2.0% of sr__1mu sits in [10, 12) -- below the lowest requested bin;
    #   * 13000 as the top edge, beyond any physical muon pt, so "25+" is genuinely inclusive.
    config.add_variable(
        name="muon0_pt",
        expression="Muon.pt[:, 0]",
        null_value=EMPTY_FLOAT,
        binning=[0.0, 10.0, 12.0, 15.0, 20.0, 25.0, 13000.0],
        unit="GeV",
        x_title=r"muon $p_{T}$",
    )

    # A FINE muon pt axis, for the 2D plot "muon0_ptfine-muon0_pnscore_heavy". It exists
    # separately from muon0_pt above because the two axes answer different questions and want
    # opposite binnings: muon0_pt has the four analysis bins plus an inclusive 25-13000 bin,
    # which is right for a yield table and useless as a plot axis -- one bin would span the
    # whole frame. This one is 2 GeV wide through the trigger-threshold region and widens above,
    # so the structure around 12/15/20/25 GeV and the IsoMu24 turn-on near 24 GeV is visible.
    config.add_variable(
        name="muon0_ptfine",
        expression="Muon.pt[:, 0]",
        null_value=EMPTY_FLOAT,
        binning=[float(x) for x in range(0, 62, 2)] + [65.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 200.0, 300.0],
        unit="GeV",
        x_title=r"muon $p_{T}$",
    )
