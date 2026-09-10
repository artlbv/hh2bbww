# coding: utf-8

"""
Kinematic classifier inputs for the TOPO trigger S/B study, as columnflow variables.

Sixteen derived quantities of the bb system, the mu+MET (W-candidate) vector and the
hadronic-W candidate, so the columnflow side can plot the same quantities that the
sensitivity study uses on the full c24v15 cocktail.

WHY CALLABLE EXPRESSIONS AND NOT A PRODUCER
Every input below is built from Jet, Muon and PuppiMET, all of which survive
ReduceEvents. Computing them in the variable expression keeps this a pure histogram
pass; a producer would mean re-running cf.ProduceColumns over all 50 datasets.

B-JET DEFINITION -- DO NOT SUBSTITUTE hbw's Bjet COLLECTION
The reference definition ranks jets by ``btagPNetB`` and takes the top two
(``ak.argsort``). That is a RANKING, not a working-point cut, and every event therefore
has a "bb system". hbw's Bjet is a WP selection on ``btagUParTAK4B`` and would give a
different, event-dependent multiplicity. ``btagPNetB`` is in the reduced files, so we
rank on it.

KNOWN DIFFERENCE FROM THE REFERENCE DEFINITION, deliberate and it affects the numbers:
the reference jet collection cleans against muons only, pT > 25, |eta| < 2.5, jet ID
off. hbw's reduced Jet is |eta| < 2.4, jet ID on, and cleaned against electrons too.
These are shape comparisons, not reproductions of the reference values. Most visible in
dR(mu, nearest clean jet) and N_jets.

A SEVENTEENTH INPUT IS DELIBERATELY ABSENT
The muon energy fraction of the nearest jet in the UNCLEANED collection is not
implemented. hbw's reduced Jet is cleaned, so the uncleaned jets are gone, and the muEF
field is not kept either. Computed against cleaned jets the quantity is degenerate and
measures nothing, so it needs a reduction change rather than an approximation.

NaN CONVENTIONS:
 - m(jj) and m(bjj) are undefined when the event has fewer than two non-b jets. That is
   information, not a defect. Histograms cannot take NaN, so they are set to EMPTY_FLOAT
   and land in underflow, where the undefined fraction stays readable rather than being
   silently folded into the first real bin.
 - ``minv`` treats an absent object as zero. That is right for the m(HH)-like sum and
   wrong for m(jj), which is why the latter is guarded with ``have2``.

ON THE UNDEFINED FRACTION -- DO NOT "FIX" IT
Because b jets are the top two by btagPNetB RANKING and not a working-point cut,
"fewer than two non-b jets" reduces to "fewer than four jets in the event": the b-tag
requirement drops out entirely. Both preselections require >=3 jets, so the undefined
fraction is exactly the fraction of 3-jet events on either side, directly comparable.
hbw's jet collection is strictly tighter, so it finds strictly fewer jets per event,
hence strictly more 3-jet events, hence a strictly HIGHER undefined fraction than the
reference. That direction is the only one allowed; a lower value would mean the port is
wrong. Measured here: 36.4%.
"""

from __future__ import annotations

import functools

import law
import order as od

from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# binning, matching the reference definition of each quantity
HT_EDGES = [80, 120, 150, 180, 210, 250, 300, 360, 440, 560, 750, 1000.]
MBB_EDGES = [0, 40, 70, 90, 105, 120, 135, 150, 170, 200, 240, 300, 400.]
MTW_EDGES = [0, 20, 40, 60, 70, 80, 90, 100, 120, 140, 170, 200.]
BPT1_EDGES = [25, 35, 45, 60, 80, 100, 130, 170, 220, 300, 400.]
DRBB_EDGES = [0, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 5.0]
MHH_EDGES = [150, 250, 300, 350, 400, 450, 500, 600, 700, 850, 1000, 1200, 1500.]
MTBW_EDGES = [0, 40, 70, 100, 130, 160, 190, 220, 260, 320, 400, 500.]
MTWW_EDGES = [0, 40, 70, 100, 130, 160, 200, 250, 310, 380, 460, 560, 700.]
DPHI_EDGES = [0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, float(np.pi)]
PTBB_EDGES = [0, 40, 70, 100, 130, 170, 210, 260, 320, 400, 500.]
DRMUB_EDGES = [0, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.4, 2.8, 3.3, 4.0]
NJET_EDGES = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 10.5]
DRMUJ_EDGES = [0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2]
MJJ_EDGES = [0, 20, 40, 60, 75, 90, 110, 135, 165, 200, 250.]
MBJJ_EDGES = [0, 60, 100, 140, 170, 200, 240, 290, 350, 430, 550.]
PUPPIMET_EDGES = [0, 20, 40, 60, 80, 100, 130, 160, 200, 250, 320.]

# every expression below reads only these
_INPUTS = {"Jet.{pt,eta,phi,btagPNetB}", "Muon.{pt,eta,phi}", "PuppiMET.{pt,phi}"}


def _f(x):
    """firsts -> float numpy, missing entries as nan"""
    return ak.to_numpy(ak.fill_none(ak.firsts(x), np.nan)).astype(np.float64)


def _adphi(d):
    """wrap to [0, pi]"""
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def _minv(objs):
    """Massless four-vector sum; NaN (absent object) contributes zero.

    Same construction used for every mass-like quantity below.
    """
    z = [(np.nan_to_num(pt), np.nan_to_num(eta), np.nan_to_num(phi)) for pt, eta, phi in objs]
    Es = sum(pt * np.cosh(eta) for pt, eta, _ in z)
    px = sum(pt * np.cos(phi) for pt, _, phi in z)
    py = sum(pt * np.sin(phi) for pt, _, phi in z)
    pz = sum(pt * np.sinh(eta) for pt, eta, _ in z)
    return np.sqrt(np.clip(Es ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0, None))


def _parts(events):
    """The shared decomposition: bb system (btag-ranked), the two leading non-b jets,
    the muon and MET. Every feature below is a projection of this."""
    jets = events.Jet
    ibb = ak.argsort(jets.btagPNetB, axis=1, ascending=False)
    bjs = jets[ibb]
    p1t, e1, f1 = _f(bjs[:, 0:1].pt), _f(bjs[:, 0:1].eta), _f(bjs[:, 0:1].phi)
    p2t, e2, f2 = _f(bjs[:, 1:2].pt), _f(bjs[:, 1:2].eta), _f(bjs[:, 1:2].phi)

    rest = bjs[:, 2:]
    qs = rest[ak.argsort(rest.pt, axis=1, ascending=False)][:, :2]
    q1 = (_f(qs[:, 0:1].pt), _f(qs[:, 0:1].eta), _f(qs[:, 0:1].phi))
    q2 = (_f(qs[:, 1:2].pt), _f(qs[:, 1:2].eta), _f(qs[:, 1:2].phi))

    mu_pt, mu_eta, mu_phi = _f(events.Muon[:, 0:1].pt), _f(events.Muon[:, 0:1].eta), _f(events.Muon[:, 0:1].phi)
    met = ak.to_numpy(events.PuppiMET.pt).astype(np.float64)
    met_phi = ak.to_numpy(events.PuppiMET.phi).astype(np.float64)

    # the mu+MET (W-candidate) transverse vector
    wx = mu_pt * np.cos(mu_phi) + met * np.cos(met_phi)
    wy = mu_pt * np.sin(mu_phi) + met * np.sin(met_phi)
    w_phi = np.arctan2(wy, wx)

    return dict(jets=jets, b1=(p1t, e1, f1), b2=(p2t, e2, f2), q1=q1, q2=q2,
                mu=(mu_pt, mu_eta, mu_phi), met=met, met_phi=met_phi, w_phi=w_phi,
                have2=np.isfinite(q1[0]) & np.isfinite(q2[0]))


def _clean(v):
    """NaN/inf -> EMPTY_FLOAT so undefined events land in underflow instead of
    poisoning the fill."""
    return np.where(np.isfinite(v), v, EMPTY_FLOAT)


# --- the individual features, each a projection of _parts -------------------------

def _mbb(events):
    P = _parts(events)
    return _clean(_minv([P["b1"], P["b2"]]))


def _dr_bb(events):
    P = _parts(events); (_, e1, f1), (_, e2, f2) = P["b1"], P["b2"]
    return _clean(np.hypot(e1 - e2, _adphi(f1 - f2)))


def _pt_bb(events):
    P = _parts(events); (p1, _, f1), (p2, _, f2) = P["b1"], P["b2"]
    x = np.nan_to_num(p1) * np.cos(np.nan_to_num(f1)) + np.nan_to_num(p2) * np.cos(np.nan_to_num(f2))
    y = np.nan_to_num(p1) * np.sin(np.nan_to_num(f1)) + np.nan_to_num(p2) * np.sin(np.nan_to_num(f2))
    return _clean(np.hypot(x, y))


def _mt_muw(events):
    """mT(mu, MET), massless: (sum pT)^2 - |sum pT_vec|^2"""
    P = _parts(events); mu_pt, _, mu_phi = P["mu"]; met, met_phi = P["met"], P["met_phi"]
    et = mu_pt + met
    px = mu_pt * np.cos(mu_phi) + met * np.cos(met_phi)
    py = mu_pt * np.sin(mu_phi) + met * np.sin(met_phi)
    return _clean(np.sqrt(np.clip(et ** 2 - px ** 2 - py ** 2, 0.0, None)))


def _dphi_muw(events):
    P = _parts(events); _, _, mu_phi = P["mu"]
    return _clean(_adphi(mu_phi - P["w_phi"]))


def _dphi_bw_min(events):
    P = _parts(events); (_, _, f1), (_, _, f2) = P["b1"], P["b2"]
    return _clean(np.minimum(_adphi(f1 - P["w_phi"]), _adphi(f2 - P["w_phi"])))


def _dphi_bbw(events):
    P = _parts(events); (p1, _, f1), (p2, _, f2) = P["b1"], P["b2"]
    x = np.nan_to_num(p1) * np.cos(np.nan_to_num(f1)) + np.nan_to_num(p2) * np.cos(np.nan_to_num(f2))
    y = np.nan_to_num(p1) * np.sin(np.nan_to_num(f1)) + np.nan_to_num(p2) * np.sin(np.nan_to_num(f2))
    return _clean(_adphi(np.arctan2(y, x) - P["w_phi"]))


def _drmin_mub(events):
    P = _parts(events); (_, e1, f1), (_, e2, f2) = P["b1"], P["b2"]; _, mu_eta, mu_phi = P["mu"]
    r1 = np.hypot(mu_eta - e1, _adphi(mu_phi - f1))
    r2 = np.hypot(mu_eta - e2, _adphi(mu_phi - f2))
    return _clean(np.minimum(r1, r2))


def _mt_bmuw(events):
    """leptonic-top candidate: the btag-pair jet CLOSEST in dphi to mu+MET, + mu + MET"""
    P = _parts(events); (p1, _, f1), (p2, _, f2) = P["b1"], P["b2"]
    mu_pt, _, mu_phi = P["mu"]; met, met_phi = P["met"], P["met_phi"]
    close1 = _adphi(f1 - P["w_phi"]) <= _adphi(f2 - P["w_phi"])
    bpt, bphi = np.where(close1, p1, p2), np.where(close1, f1, f2)
    et = bpt + mu_pt + met
    px = bpt * np.cos(bphi) + mu_pt * np.cos(mu_phi) + met * np.cos(met_phi)
    py = bpt * np.sin(bphi) + mu_pt * np.sin(mu_phi) + met * np.sin(met_phi)
    return _clean(np.sqrt(np.clip(et ** 2 - px ** 2 - py ** 2, 0.0, None)))


def _mt_ww(events):
    """mu + MET + the two leading non-b jets, transverse"""
    P = _parts(events); mu_pt, _, mu_phi = P["mu"]; met, met_phi = P["met"], P["met_phi"]
    terms = [(mu_pt, mu_phi), (met, met_phi),
             (np.nan_to_num(P["q1"][0]), np.nan_to_num(P["q1"][2])),
             (np.nan_to_num(P["q2"][0]), np.nan_to_num(P["q2"][2]))]
    et = sum(pt for pt, _ in terms)
    px = sum(pt * np.cos(phi) for pt, phi in terms)
    py = sum(pt * np.sin(phi) for pt, phi in terms)
    return _clean(np.sqrt(np.clip(et ** 2 - px ** 2 - py ** 2, 0.0, None)))


def _mhh(events):
    """bb + mu + MET + hadronic-W candidate; absent jets contribute zero by design"""
    P = _parts(events); mu_pt, mu_eta, mu_phi = P["mu"]
    met4 = (P["met"], np.zeros(len(P["met"])), P["met_phi"])
    return _clean(_minv([P["b1"], P["b2"], (mu_pt, mu_eta, mu_phi), met4, P["q1"], P["q2"]]))


def _mjj(events):
    """hadronic-W candidate; NaN (-> EMPTY_FLOAT) when fewer than two non-b jets"""
    P = _parts(events)
    return _clean(np.where(P["have2"], _minv([P["q1"], P["q2"]]), np.nan))


def _mbjj(events):
    """hadronic-top candidate, built with the b jet FARTHER from the muon"""
    P = _parts(events); (p1, e1, f1), (p2, e2, f2) = P["b1"], P["b2"]; _, mu_eta, mu_phi = P["mu"]
    r1 = np.hypot(mu_eta - e1, _adphi(mu_phi - f1))
    r2 = np.hypot(mu_eta - e2, _adphi(mu_phi - f2))
    far1 = r1 > r2
    bf = (np.where(far1, p1, p2), np.where(far1, e1, e2), np.where(far1, f1, f2))
    return _clean(np.where(P["have2"], _minv([bf, P["q1"], P["q2"]]), np.nan))


def _drmuj(events):
    """dR to the nearest CLEANED jet. See the module docstring: hbw's jet definition
    differs from the reference, so this is a shape comparison, not the reference value."""
    P = _parts(events); jets = P["jets"]; _, mu_eta, mu_phi = P["mu"]
    d = np.sqrt((jets.eta - mu_eta) ** 2 + (_adphi(jets.phi - mu_phi)) ** 2)
    return _clean(ak.to_numpy(ak.fill_none(ak.min(d, axis=1), np.nan)).astype(np.float64))


def _n_jets(events):
    return _clean(ak.to_numpy(ak.num(events.Jet, axis=1)).astype(np.float64))


def _puppimet(events):
    return _clean(ak.to_numpy(events.PuppiMET.pt).astype(np.float64))


_SPECS = [
    ("topo_an_mhh", _mhh, MHH_EDGES, r"$m(bb+jj+\mu+MET)$", "GeV"),
    ("topo_an_mt_bmuw", _mt_bmuw, MTBW_EDGES, r"$m_{T}(b_{close}+\mu+MET)$", "GeV"),
    ("topo_an_mt_ww", _mt_ww, MTWW_EDGES, r"$m_{T}(\mu+MET+jj)$", "GeV"),
    ("topo_an_dphi_bw_min", _dphi_bw_min, DPHI_EDGES, r"$\min_{b}\Delta\phi(b,\mu+MET)$", ""),
    ("topo_an_dr_bb", _dr_bb, DRBB_EDGES, r"$\Delta R(b_{1},b_{2})$", ""),
    ("topo_an_mbb", _mbb, MBB_EDGES, r"$m_{bb}$", "GeV"),
    ("topo_an_mt_muw", _mt_muw, MTW_EDGES, r"$m_{T}(\mu,MET)$", "GeV"),
    ("topo_an_dphi_muw", _dphi_muw, DPHI_EDGES, r"$\Delta\phi(\mu,\mu+MET)$", ""),
    ("topo_an_pt_bb", _pt_bb, PTBB_EDGES, r"$p_{T}(bb)$", "GeV"),
    ("topo_an_drmin_mub", _drmin_mub, DRMUB_EDGES, r"$\min_{b}\Delta R(\mu,b)$", ""),
    ("topo_an_dphi_bbw", _dphi_bbw, DPHI_EDGES, r"$\Delta\phi(bb,\mu+MET)$", ""),
    ("topo_an_n_jets", _n_jets, NJET_EDGES, r"$N_{jets}$", ""),
    ("topo_an_drmuj", _drmuj, DRMUJ_EDGES, r"$\Delta R(\mu,$ nearest clean jet$)$", ""),
    ("topo_an_mjj", _mjj, MJJ_EDGES, r"$m(jj)$, hadronic-$W$ candidate", "GeV"),
    ("topo_an_mbjj", _mbjj, MBJJ_EDGES, r"$m(bjj)$, hadronic-top candidate", "GeV"),
    ("topo_an_puppimet", _puppimet, PUPPIMET_EDGES, r"PuppiMET", "GeV"),
]

#: plot/yield-friendly comma-joined list of every variable this module adds
TOPO_AN_VARIABLES = ",".join(name for name, *_ in _SPECS)


def add_topo_an_variables(config: od.Config) -> None:
    """Register the TOPO S/B kinematic classifier inputs on *config*."""
    for name, expr, edges, x_title, unit in _SPECS:
        if name in config.variables:
            continue
        config.add_variable(
            name=name,
            expression=expr,
            null_value=EMPTY_FLOAT,
            binning=list(map(float, edges)),
            unit=unit or None,
            x_title=x_title,
            aux={"inputs": set(_INPUTS)},
        )
    logger.debug(f"added {len(_SPECS)} topo kinematic variables to config {config.name}")
