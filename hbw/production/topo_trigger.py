# coding: utf-8

"""
Offline feature production for the TOPO trigger sensitivity study.

The TOPO path does not exist in the 2024 HLT menu, so its per-event decision cannot be read from
Summer24 NanoAODv15 and has to be emulated. The emulation consumes the *offline* feature vector
the trigger-efficiency estimator was fit on, as fixed by the reference implementation.

This module mirrors that definition. It deliberately does NOT reuse hbw's own object selections:

* hbw's ``events.Jet`` after ``cf.ReduceEvents`` is the *selected* collection -- ``pt >= 25``,
  ``|eta| <= 2.4``, tight jet ID, cross-cleaned against selected electrons *and* muons
  (:py:mod:`hbw.selection.jet`). The estimator was fit on ``|eta| < 2.5``, **no jet ID**, cleaned
  against muons only. The 2.4-vs-2.5 and the electron cleaning move HT by several percent, in
  exactly the turn-on region this study lives in.
* hbw's ``Muon.is_tight`` uses the working-point isolation flag on ``pfRelIso04_all``
  (:py:mod:`hbw.selection.lepton`); the estimator uses a raw ``pfRelIso03_all < 0.15`` cut.
* hbw's b-tag counting uses ``btagUParTAK4B`` at the medium WP for ``year >= 2024``; the estimator
  counts ``btagPNetB > 0.2``. Both coexist -- ``btagPNetB`` is separately stored and kept.

Because raw NanoAOD jets are unrecoverable after reduction, this producer must run from inside
``cf.SelectEvents``. Its columns then persist through ``ColumnCollection.ALL_FROM_SELECTOR``.

On "PF HT": ``extract.py`` defines feature 3 as the *offline* muon-cross-cleaned
``HT (sum jet pt > 25, |eta| < 2.5)``. The ``PFHT150`` in the reference path's name is what that
offline HT is a proxy for -- the mismatch is studied separately and is not a bug here.
"""

from __future__ import annotations

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column, optional_column as optional

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


#: L1 total-HT thresholds whose seeds are stored in NanoAOD. The topo repo measured that
#: ``L1_HTT280er`` fires iff the stored L1 HT is >= 280 (100.00% agreement, see
#: ``exploratory/explore_l1ht_electron_contamination.py``), so the OR of these bits gives a
#: staircase proxy for the continuous L1 HT that NanoAOD does not store.
L1HT_THRESHOLDS = (120.0, 150.0, 200.0, 255.0, 280.0, 320.0, 360.0)

#: feature column order. Load-bearing: it is fixed by the exporter's ``ALL_FEATURES`` and must
#: match the order the estimator was fit on. Never reorder without re-exporting the model.
#:
#: v5 (bundle ``topo_ens_K20_v2c_pnet``): the six original columns stay in place at indices 0-5 and
#: the isolation/ID block plus the four muon PNet scores are appended, so an estimator picks the
#: columns it was fitted on through its own ``features`` index list. No estimator uses all fifteen:
#: the muon-only legs take eleven (no ht/lead_btag/lead_bpt) and ``mu_iso`` is a preselection cut
#: rather than a feature in v2, so index 2 is carried but unused. See :py:data:`MU_FEATURES`.
FEATURE_ORDER = (
    "mu_pt", "mu_eta", "mu_iso", "ht", "lead_btag", "lead_bpt",
    "tkRelIso", "jetRelIso", "miniPFRelIso_all", "sip3d", "jetDF",
    "pnScore_prompt", "pnScore_heavy", "pnScore_light", "pnScore_tau",
)

#: the per-muon columns of :py:data:`FEATURE_ORDER` that are read straight off the selected muon.
#: ``jetDF`` IS a flat per-muon NanoAOD branch (``Muon_jetDF``, the DeepJet b+bb+lepb sum of the
#: muon's associated uncleaned jet, 0 where there is none) -- see the note in ``config_run2.py``
#: next to the keep_columns entry; it does not have to be rebuilt from ``Muon.jetIdx``.
MU_FEATURES = (
    "tkRelIso", "jetRelIso", "miniPFRelIso_all", "sip3d", "jetDF",
    "pnScore_prompt", "pnScore_heavy", "pnScore_light", "pnScore_tau",
)

#: the columns of :py:data:`FEATURE_ORDER` that describe the EVENT rather than the muon. These are
#: the ones only ``topo_features`` can build (they need the cleaned jet collection), so they are
#: always read from the stored ``topo_feat`` field; everything else can be re-read off ``Muon``.
EVENT_FEATURES = ("mu_pt", "mu_eta", "mu_iso", "ht", "lead_btag", "lead_bpt")

#: feature name -> ``Muon`` branch, where the two differ.
MU_SOURCE = {"mu_pt": "pt", "mu_eta": "eta", "mu_iso": "pfRelIso03_all"}

#: ``Muon.jetRelIso`` carries -1.0 for a muon with no associated jet (0.32% of the training rows).
#: That is a REAL trained-on value, not a sentinel to be repaired: the exporter kept it and so must
#: this producer. It is the one feature whose legitimate range dips below zero, which is why the
#: evaluable mask below tests against EMPTY_FLOAT rather than against a sign.
JETRELISO_NO_JET = -1.0


@producer(
    uses={
        "Jet.{pt,eta,phi,mass,btagPNetB}",
        "Muon.{pt,eta,phi,mass,pfRelIso03_all,tightId}",
        "Muon.{tkRelIso,jetRelIso,miniPFRelIso_all,sip3d,jetDF}",
        "Muon.{pnScore_prompt,pnScore_heavy,pnScore_light,pnScore_tau}",
    } | {
        optional(f"L1.HTT{int(t)}er") for t in L1HT_THRESHOLDS
    },
    produces={
        f"topo_feat.{f}" for f in FEATURE_ORDER
    } | {
        "topo_feat.{n_btag_pnet,valid,l1ht_proxy}",
    },
    # configurable attributes, defaulting to the reference object selection
    jet_pt=25.0,
    jet_eta=2.5,
    jet_id=False,
    mu_pt=10.0,
    mu_eta=2.4,
    mu_iso=0.15,
    btag_column="btagPNetB",
    btag_wp=0.2,
    clean_dr=0.4,
    n_jet_presel=3,
    version=0,
)
def topo_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces the ``topo_feat`` field holding the six estimator features, the PNet b-tag
    multiplicity, the training-support flag and the L1 HT staircase proxy.
    """
    # --- muon selection: raw cuts, mirroring extract.build() ---
    mu_sel = (
        events.Muon.tightId &
        (events.Muon.pfRelIso03_all < self.mu_iso) &
        (events.Muon.pt > self.mu_pt) &
        (abs(events.Muon.eta) < self.mu_eta)
    )
    mu_g = events.Muon[mu_sel]
    n_mu = ak.num(mu_g, axis=1)

    # --- jet selection: acceptance, (no) jet ID, then muon cross-cleaning, in that order ---
    jet_mask = (events.Jet.pt > self.jet_pt) & (abs(events.Jet.eta) < self.jet_eta)
    if self.jet_id:
        jet_mask = jet_mask & (events.Jet.jetId & 2 == 2)
    jet_acc = events.Jet[jet_mask]

    # remove any jet within clean_dr of ANY selected muon; events without a selected muon have
    # nothing to clean against, hence the fill_none sentinel
    min_dr = ak.fill_none(ak.min(jet_acc.metric_table(mu_g), axis=-1), 999.0)
    jet_g = jet_acc[min_dr >= self.clean_dr]
    n_jet = ak.num(jet_g, axis=1)

    # --- leading isolated muon ---
    lead_mu = mu_g[ak.argmax(mu_g.pt, axis=1, keepdims=True)]
    events = set_ak_column(events, "topo_feat.mu_pt", _first(lead_mu.pt))
    events = set_ak_column(events, "topo_feat.mu_eta", _first(lead_mu.eta))
    events = set_ak_column(events, "topo_feat.mu_iso", _first(lead_mu.pfRelIso03_all))

    # --- per-muon isolation / ID / PNet block (v5). Read off the SAME leading muon, so every
    #     feature in a row describes one object; ``_first`` fills EMPTY_FLOAT where there is none.
    for f in MU_FEATURES:
        events = set_ak_column(events, f"topo_feat.{f}", _first(lead_mu[f]))


    # --- jet-level features ---
    btag = jet_g[self.btag_column]
    lead_b = jet_g[ak.argmax(btag, axis=1, keepdims=True)]
    events = set_ak_column(events, "topo_feat.ht", ak.sum(jet_g.pt, axis=1))
    events = set_ak_column(events, "topo_feat.lead_btag", _first(lead_b[self.btag_column]))
    events = set_ak_column(events, "topo_feat.lead_bpt", _first(lead_b.pt))
    events = set_ak_column(events, "topo_feat.n_btag_pnet", ak.sum(btag > self.btag_wp, axis=1))

    # --- training support. The estimator was never fit outside its own preselection, so events
    #     failing this must not be reweighted; the emulation falls back to the stored OR2 decision
    #     there and reports the fraction.
    #
    #     CAUTION -- ``valid`` is WEAKER than the estimator's training preselection, and is not on
    #     its own a licence to evaluate the model. The reference preselection is
    #     ``n_mu >= 1 & n_jet >= 3 & n_btag_pnet >= 2 & n_tight_electrons == 0``; only the first two
    #     terms are checked here, because the other two are not properties of the feature vector.
    #     Measured on reduced 2024 signal, ``valid`` holds for 57.5% of events while the full
    #     preselection holds for 34.4% -- so treating ``valid`` as "in support" would extrapolate
    #     the model over 40% of the events it selects, silently.
    #
    #     Any consumer that feeds these columns to a model must therefore AND in the missing terms;
    #     :py:func:`topo_or3_weights` does this via its ``min_n_btag_pnet`` attribute and writes the
    #     result as ``topo_in_support``. Prefer that column over re-deriving the mask.
    #
    #     This matters more than it looks because the sentinel below is EMPTY_FLOAT, i.e. finite:
    #     nothing NaN-based will catch it, and XGBoost will read it as a real, very negative
    #     feature value rather than as missing. Plotting the columns is unaffected -- undefined
    #     events land in underflow and the fraction stays readable. ---
    events = set_ak_column(
        events,
        "topo_feat.valid",
        (n_mu >= 1) & (n_jet >= self.n_jet_presel),
    )

    # --- L1 HT staircase proxy (7th feature; see the module docstring) ---
    events = set_ak_column(events, "topo_feat.l1ht_proxy", l1ht_proxy(events))

    return events


def _first(x: ak.Array) -> ak.Array:
    """
    First entry per event, or EMPTY_FLOAT where the collection is empty.

    ``extract.py`` uses NaN here, but it only ever evaluates features on rows that already passed
    its preselection. This producer writes a column for *every* event, and ``cf.SelectEvents`` calls
    ``raise_if_not_finite`` on its output, so a NaN is a hard error rather than a marker. The
    sentinel is safe because it is only reachable where ``topo_feat.valid`` is False, and no event
    with ``valid == False`` may be reweighted -- the emulation falls back to the stored OR2 decision
    there. Anything that consumes these features must check ``valid`` first.
    """
    return ak.fill_none(ak.firsts(x), EMPTY_FLOAT)


def l1ht_proxy(events: ak.Array) -> ak.Array:
    """
    Highest L1 HT threshold whose stored seed fired, or 0 if none did.

    NanoAOD does not store ``L1EtSum``, so the continuous L1 HT that the estimator's 7th feature
    uses is unavailable. What it does store are the seed bits, and the topo repo measured that
    ``L1_HTT280er`` fires iff the L1 HT is >= 280 (100.00% agreement), so their OR is an exact
    8-level quantisation of the same quantity.

    This is a staircase: no resolution above the highest fired threshold, and none below the lowest.
    Feeding it to a model fit on the continuous variable is a covariate shift the model is not
    calibrated for -- the estimator has to be refit on the quantised variable before this can be
    used as the 7th feature. It is produced here so that refit has an input and so the loss can be
    measured.
    """
    proxy = np.zeros(len(events), dtype=np.float32)
    if "L1" not in events.fields:
        logger.warning_once(
            "topo_no_l1_collection",
            "no L1 collection in this sample; topo_feat.l1ht_proxy is 0 everywhere",
        )
        return proxy
    available = [t for t in L1HT_THRESHOLDS if f"HTT{int(t)}er" in events.L1.fields]
    if not available:
        logger.warning_once(
            "topo_no_l1ht_seeds",
            "none of the L1 HTT*er seeds are stored; topo_feat.l1ht_proxy is 0 everywhere",
        )
        return proxy
    for t in available:
        proxy = np.where(ak.to_numpy(events.L1[f"HTT{int(t)}er"]), np.float32(t), proxy)
    return proxy


#: the reference trigger of the study. Both are stored bits in Summer24 NanoAODv15, so "OR2" is
#: available as *truth* and does not have to be emulated at all. This matters: the b-tag leg must
#: not be reweighted from offline-only variables. Only the TOPO residual is.
OR2_PATHS = ("IsoMu24", "Mu12_IsoVVL_PFHT150_PNetBTag0p53")


@producer(
    uses={f"HLT.{p}" for p in OR2_PATHS},
    produces={"topo_or2", "topo_trigger_weight_or2"},
    version=0,
)
def topo_or2_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    The reference-trigger arm of the emulation, which needs no model.

    ``w_OR2 = d_OR2 * SF_OR2(c)``, where ``d_OR2`` is the stored decision. ``SF_OR2`` is to be
    measured in this setup with :py:class:`hbw.tasks.trigger_sf.ComputeTriggerSF` against the
    MET-orthogonal reference rather than transported from elsewhere; until that measurement exists it
    is identically 1, which makes this the *uncorrected* OR2 baseline. Say so wherever it is
    plotted.
    """
    d_or2 = events.HLT[OR2_PATHS[0]]
    for p in OR2_PATHS[1:]:
        d_or2 = d_or2 | events.HLT[p]

    events = set_ak_column(events, "topo_or2", d_or2)
    events = set_ak_column(
        events,
        "topo_trigger_weight_or2",
        ak.values_astype(d_or2, np.float32),
    )
    return events


#: the single-muon reference leg on its own. This is the trigger the analysis *already* has, so it
#: is the baseline the TOPO path has to beat: OR2/IsoMu24 is the acceptance the second leg buys,
#: whereas OR2/none only says how far OR2 is from the ceiling.
ISOMU_PATH = "IsoMu24"


@producer(
    uses={f"HLT.{ISOMU_PATH}"},
    produces={"topo_isomu24", "topo_trigger_weight_isomu24"},
    version=0,
)
def topo_isomu24_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    The single-muon-only arm: ``w = d_IsoMu24``, the stored decision of ``HLT_IsoMu24``.

    Like :py:func:`topo_or2_weights` this needs no emulation -- the bit is stored in Summer24
    NanoAODv15 -- and carries no scale factor yet, so it is the *uncorrected* IsoMu24 baseline.
    Its purpose is the OR2-vs-IsoMu24 comparison: the gain from adding
    ``Mu12_IsoVVL_PFHT150_PNetBTag0p53`` to the existing single-muon path.
    """
    d_isomu = events.HLT[ISOMU_PATH]

    events = set_ak_column(events, "topo_isomu24", d_isomu)
    events = set_ak_column(
        events,
        "topo_trigger_weight_isomu24",
        ak.values_astype(d_isomu, np.float32),
    )
    return events


#: names of the estimators carried in the exported bundle, in the order they are reported.
#: ``isomu24``, ``mu12`` and ``or2`` all have stored truth in Summer24 NanoAODv15 and are emulated
#: here ONLY so the emulation can be checked against that truth -- they are the closure test, not
#: an input to any weight. Only ``topo_res`` is load-bearing, because TOPO is the one path that
#: does not exist in the 2024 menu.
#:
#: ``topo`` and ``or3`` are the marginal TOPO efficiency and the directly fitted three-path union.
#: Neither builds a weight -- the weight uses the residual, precisely so that no leg-factorisation
#: assumption enters -- but they are what a reader actually wants to see, and ``or3`` doubles as a
#: check on the residual construction: ``d_OR2 + (1 - d_OR2) * eps_res`` and the directly fitted
#: ``eps_OR3`` estimate the same quantity by different routes, so a disagreement between them is a
#: statement about the construction that nothing else in the chain would make.
TOPO_ESTIMATORS = ("isomu24", "mu12", "or2", "topo_res", "topo", "or3")

#: estimators the v2 bundle adds beyond :py:data:`TOPO_ESTIMATORS`, written when the loaded bundle
#: carries them and skipped silently when it does not, so a v1 bundle still loads.
#:
#: * CONTROL LEGS -- ``mu15``, ``mu12eta2p3``, ``mu17_trkisovvl``, ``mu15_isovvvl_ht450``: an
#:   isolation ladder (none -> tracker-VVL -> PF-VVVL -> IsoMu24) fitted for the MC-internal study
#:   of how much of the emulation's non-prompt behaviour is the online isolation working point.
#:   All four paths are PRESCALED in data, so they are never data-validated efficiencies.
#: * CONDITIONALS -- ``*_given_<leg>``: fitted and evaluated ONLY on events whose named stored bit
#:   fired. See :py:data:`TOPO_CONDITIONAL_NOTE`.
TOPO_EXTRA_ESTIMATORS = (
    "mu15", "mu15_isovvvl_ht450", "mu12eta2p3", "mu17_trkisovvl",
    "topo_given_mu12eta2p3", "or3_given_mu12eta2p3", "or2_given_mu12eta2p3",
    "topo_given_mu15", "or3_given_mu15",
)

#: why a conditional estimator may never be used as a plain weight. A conditional carries
#: ``conditioned_on = "<stored HLT column>"`` and estimates ``P(target | that bit fired, x)``; the
#: consumer must apply ``w = eps x stored_bit``. Using the UNCONDITIONAL estimator in that product
#: instead is silent and wrong -- it under-counts, by percentage points rather than fractions of
#: one, because on seeded events the true efficiency sits above its unconditional average; the
#: exporter measured the size per sample and reports it in the release notes of the bundle, which
#: is where that number belongs. The bit is PER EVENT while
#: the estimator is per in-support muon; with the ``sl1_topo`` selection there is exactly one
#: selected muon per event, so the assignment is unambiguous here and would not be in a
#: multi-muon selection.
#: stored HLT columns any conditional in the v2 bundle can be conditioned on. Declared optional in
#: ``uses`` so that a reduction predating commit 39eea63 (which carries only the original eight
#: paths) still loads; the setup below hard-fails instead if a conditional is actually requested
#: and its column is absent, because eps x (missing bit) has no safe default.
TOPO_CONDITION_COLUMNS = ("Mu12eta2p3", "Mu15")

TOPO_CONDITIONAL_NOTE = (
    "conditional estimators carry conditioned_on and must be applied as eps * stored_bit"
)

#: the estimator used by the residual construction, which is kept as a cross-check only.
TOPO_RESIDUAL_ESTIMATOR = "topo_res"

#: estimators that may be used as the applied weight. ``or3`` is the proposed menu, ``topo`` the
#: TOPO path alone; the legs are listed so a leg-only study needs no code change. ``topo_res`` is
#: deliberately absent -- it is a residual, not an efficiency, and is not a weight on its own.
TOPO_WEIGHT_CHOICES = ("or3", "topo", "or2", "isomu24", "mu12")


@producer(
    uses={topo_or2_weights, topo_isomu24_weights} | {f"topo_feat.{f}" for f in EVENT_FEATURES} | {
        "topo_feat.valid", "topo_feat.n_btag_pnet",
        # phi and mass are not features; the Muon collection carries Lorentz-vector behaviour and
        # awkward refuses to build the record without its azimuthal coordinates.
        "Muon.{pt,eta,phi,mass,pfRelIso03_all,tkRelIso,jetRelIso,miniPFRelIso_all,sip3d,jetDF}",
        "Muon.{pnScore_prompt,pnScore_heavy,pnScore_light,pnScore_tau}",
    } | {
        optional(f"topo_feat.{f}") for f in MU_FEATURES
    },
    produces={
        topo_or2_weights, topo_isomu24_weights,
        "topo_trigger_weight", "topo_trigger_weight_built", "topo_in_support",
    } | {
        f"topo_eff_{n}" for n in TOPO_ESTIMATORS + TOPO_EXTRA_ESTIMATORS
    } | {
        f"topo_eff_{n}_std" for n in TOPO_ESTIMATORS + TOPO_EXTRA_ESTIMATORS
    } | {
        f"topo_weight_{n}" for n in TOPO_EXTRA_ESTIMATORS if "_given_" in n
    } | {
        optional(f"HLT.{c}") for c in TOPO_CONDITION_COLUMNS
    },
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_topo.sh"),
    mc_only=True,
    #: minimum number of PNet b-tags. NOT cosmetic: the estimator's training preselection is
    #: ``n_mu>=1 & n_jet>=3 & n_btag_pnet>=2 & n_ele_tight==0`` while ``topo_feat.valid`` is only
    #: the first two terms, so ``valid`` alone would extrapolate the model outside its support.
    min_n_btag_pnet=2,
    #: the estimator whose ensemble mean IS the weight, one of TOPO_WEIGHT_CHOICES. See the
    #: docstring for why this is a directly fitted union rather than the residual decomposition.
    weight_estimator="or3",
    #: the muon selection used to pick the object the features describe. Must match
    #: ``topo_features``' cuts, since the two are asserted to select the same muon.
    mu_pt=10.0,
    mu_eta=2.4,
    mu_iso=0.15,
    version=5,
)
def topo_or3_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    The proposed-menu arm: one directly fitted efficiency, applied as a probability weight.

    .. code-block:: text

        w = eps_<weight_estimator>(c)     in support
        w = d_OR2                         outside it

    ``weight_estimator`` defaults to ``or3``, the three-path union ``IsoMu24 | Mu12 | TOPO``
    fitted as a single target; ``topo`` gives the TOPO path alone, for the menu question that
    drops the existing legs entirely.

    **Why a single directly fitted target and not the residual decomposition.** The exact identity
    ``P(OR3) = P(OR2) + P(TOPO & ~OR2)`` lets the stored OR2 bit carry the first, dominant term
    and a model carry only the small remainder, which is attractive on paper. It is the wrong
    trade here, for two reasons that both point the same way:

    * **Scale factors.** The deliverable applies data efficiencies, so the residual form needs
      ``SF_OR2`` on a term worth ~86% of the weight -- and ``SF_OR2`` carries the
      ``Mu12_IsoVVL_PFHT150_PNetBTag0p53`` leg, whose online b-tag requirement is the part of the
      menu with the widest expected SF spread. In a directly fitted ``or3`` that leg enters only
      where it is the *sole* path firing, which is a small corner: TOPO alone already reaches
      92.4% against 93.3% for the union.
    * **Closure.** The residual form divides by ``1 - eps_OR2``, so it imports the OR2 emulation
      into a denominator -- and ``mu12`` is measured to be the worst-closing of the three targets
      that have a stored bit (+2.96 +- 0.76 pp, 3.9 sigma, against +0.99 for ``or2`` and -0.46 for
      ``isomu24``). A directly fitted ``or3`` does not use ``eps_mu12`` at all.

    Neither route assumes leg factorisation: ``or3`` is fitted on the union bit itself, exactly as
    ``topo_res`` is fitted on ``TOPO & ~OR2``. The inclusion-exclusion form that *would* assume it
    is not used anywhere.

    The residual construction is still computed, as ``topo_trigger_weight_built``, because the two
    routes estimate the same quantity by independent paths and their difference is the sharpest
    available statement about the emulation -- on 2024 signal they agree to 0.26 pp. It is a
    cross-check, not the weight.

    **A probability weight, not a decision.** ``w = eps``, not ``w = 1{eps > 0.5}``. A threshold is
    simply biased (``E[1{eps>0.5}] != E[eps]``) and a Bernoulli draw throws away precision for
    nothing; the probability weight is unbiased in yield and correct in shape.

    Outside the training support the weight falls back to the stored ``d_OR2``. For ``or3`` that is
    a floor and not an extrapolation, since ``OR3`` contains ``OR2`` by construction; for ``topo``
    it is neither a bound nor an estimate, so out-of-support events must be cut on
    ``topo_in_support`` rather than trusted. The fallback exists so the column is finite.

    Besides the weight this writes ``topo_eff_<name>`` and ``topo_eff_<name>_std`` for every
    estimator in the bundle. Three of the six have a stored decision in Summer24 NanoAODv15 and are
    emulated *only* so the emulation can be checked against it: ``topo_eff_isomu24`` pairs with
    ``topo_isomu24``, ``topo_eff_or2`` with ``topo_or2``, and ``topo_eff_mu12`` with the stored HLT
    bit.

    **The eps columns are not gated on the support (v4).** They carry a prediction wherever the
    six features are real -- including outside ``topo_in_support``, where it is an extrapolation
    the flag marks -- and ``EMPTY_FLOAT`` only where a feature is a sentinel and no prediction
    exists. Until v3 they were 0.0 outside the support, which is indistinguishable from a real
    "efficiency is ~0" and silently biased any mean taken over a wider selection. Mask on
    ``topo_eff_<name> != EMPTY_FLOAT`` before averaging, and on ``topo_in_support`` as well if
    extrapolated rows are unwanted. The WEIGHTS are unaffected by this and remain gated on the
    support with the ``d_OR2`` fallback.
    """
    events = self[topo_or2_weights](events, **kwargs)
    # the stored IsoMu24 decision travels alongside, so a single pass writes each emulated
    # efficiency next to its own truth column on the same events -- the closure test is then
    # reproducible from this output alone, with no second pass and no join.
    events = self[topo_isomu24_weights](events, **kwargs)

    feat = events.topo_feat
    in_support = np.asarray(
        ak.to_numpy(feat.valid & (feat.n_btag_pnet >= self.min_n_btag_pnet)),
        dtype=bool,
    )
    events = set_ak_column(events, "topo_in_support", in_support)

    # WHERE THE MODEL IS EVALUATED, AND WHY THAT IS NOT THE SUPPORT (changed in v4).
    #
    # Until v3 the boosters saw in-support rows only and ``topo_eff_*`` was filled with 0.0
    # everywhere else. That zero is indistinguishable from a genuine "efficiency is ~0"
    # prediction and it is finite, so no NaN check catches it: averaging ``topo_eff_or2`` over
    # any selection wider than the support silently divided the answer by the support fraction
    # (measured on tt FH: 0.043 instead of 0.677, a factor 16 low, with no warning). An
    # efficiency column must never read as zero merely because a flag is off.
    #
    # The precondition for evaluating is NOT membership of the support -- it is that the six
    # features are real. Those are different questions, and on tt FH they split three ways:
    #
    #     in support (valid & n_btag_pnet >= 2)   6.3%   all features real
    #     valid but n_btag_pnet < 2              20.5%   all features real, ZERO sentinels
    #     not valid (no muon / < 3 jets)         73.1%   mu_pt/eta/iso are EMPTY_FLOAT in 99.99%
    #
    # The middle band was being zeroed purely because of a b-tag count while carrying six
    # perfectly real features; the model has something to say there and it is an extrapolation
    # in n_btag, which is exactly what ``topo_in_support`` is for flagging. The bottom band is
    # a different thing entirely: with no muon there is no mu_pt, and feeding EMPTY_FLOAT to
    # XGBoost does not extrapolate -- it reads -99999 as a real, very negative feature value
    # and returns a confident number that means nothing.
    #
    # So: evaluate wherever no feature is a sentinel, and mark the rest EMPTY_FLOAT rather than
    # 0.0. EMPTY_FLOAT is this codebase's "not defined" convention, variables already filter it
    # via ``null_value``, and unlike 0.0 an unmasked mean over it is unmissably wrong instead of
    # plausibly wrong. Consumers get three distinguishable states:
    #
    #     topo_eff_* == EMPTY_FLOAT                no prediction exists (no muon to trigger on)
    #     topo_eff_* != EMPTY_FLOAT, ~in_support   a real prediction, EXTRAPOLATED -- check the bit
    #     topo_in_support                          a real prediction inside the training support
    # FEATURE MATRIX (v5). The six event/muon-kinematic columns come from ``topo_feat``, written
    # at reduction time. The isolation/ID/PNet block does NOT: ``topo_features`` gained those
    # columns in v5 and reductions made before that carry only the original six, so they are read
    # here off the reduced ``Muon`` collection instead, which keeps every one of them. When a
    # reduction DOES carry them the stored column wins, so a future re-reduction changes nothing.
    #
    # The muon must be the same one ``topo_feat`` described, or a row would mix two objects. The
    # reduced collection is already tight-ID'd and pT > 10 (sl1_topo's veto_mu_mask), so the
    # study's selection reduces to the isolation and eta cuts; the agreement of pt/eta/iso with
    # ``topo_feat`` is asserted per row below and a disagreeing row is simply not evaluated.
    mu = events.Muon
    mu_g = mu[(mu.pfRelIso03_all < self.mu_iso) & (mu.pt > self.mu_pt) & (abs(mu.eta) < self.mu_eta)]
    lead_mu = mu_g[ak.argmax(mu_g.pt, axis=1, keepdims=True)]

    def _mu_col(name: str) -> np.ndarray:
        if name in feat.fields:
            return np.asarray(ak.to_numpy(feat[name]), dtype=np.float64)
        return np.asarray(
            ak.to_numpy(ak.fill_none(ak.firsts(lead_mu[MU_SOURCE.get(name, name)]), EMPTY_FLOAT)),
            dtype=np.float64,
        )

    if len(in_support):
        cols = []
        for f in FEATURE_ORDER:
            cols.append(
                np.asarray(ak.to_numpy(feat[f]), dtype=np.float64) if f in EVENT_FEATURES
                else _mu_col(f)
            )
        x_all = np.column_stack(cols)
        # same-object check on the three columns that exist in both places
        mism = np.zeros(len(in_support), dtype=bool)
        for f, src in (("mu_pt", "pt"), ("mu_eta", "eta"), ("mu_iso", "pfRelIso03_all")):
            stored = np.asarray(ak.to_numpy(feat[f]), dtype=np.float64)
            here = np.asarray(
                ak.to_numpy(ak.fill_none(ak.firsts(lead_mu[src]), EMPTY_FLOAT)), dtype=np.float64,
            )
            both = (stored != EMPTY_FLOAT) & (here != EMPTY_FLOAT)
            mism |= both & ~np.isclose(stored, here, rtol=0.0, atol=1e-3)
        if mism.any():
            logger.warning(
                f"{int(mism.sum())} of {len(mism)} rows select a different leading muon here than "
                f"topo_feat did; they are not evaluated (eps = EMPTY_FLOAT)",
            )
    else:
        x_all = np.zeros((0, len(FEATURE_ORDER)))
        mism = np.zeros(0, dtype=bool)
    # ``mu_iso`` (index 2) is carried by FEATURE_ORDER but used by no v2 estimator, and a row may
    # legitimately hold EMPTY_FLOAT in a column nothing reads. Require realness only of the columns
    # some estimator actually indexes. ``jetRelIso == -1.0`` is a trained-on value, not a sentinel.
    used = sorted({i for idx, _ in self.topo_members.values() for i in idx}) if len(x_all) else []
    evaluable = (
        np.all(x_all[:, used] != EMPTY_FLOAT, axis=1) & np.all(np.isfinite(x_all[:, used]), axis=1) & ~mism
        if len(x_all) else np.zeros(0, dtype=bool)
    )
    x = x_all[evaluable]

    n_out = int((~in_support).sum())
    if n_out:
        n_extrap = int((evaluable & ~in_support).sum())
        logger.info(
            f"{n_out} of {len(in_support)} events ({n_out / len(in_support) * 100:.2f}%) are "
            f"outside the estimator's training support; the applied weight falls back to "
            f"w = d_OR2 there. Of those, {n_extrap} carry real features and DO receive an "
            f"extrapolated eps (flagged by topo_in_support == False); the remaining "
            f"{n_out - n_extrap} have a sentinel feature and get eps = EMPTY_FLOAT",
        )

    for name in self.topo_members:
        mean = np.full(len(in_support), EMPTY_FLOAT, dtype=np.float32)
        std = np.full(len(in_support), EMPTY_FLOAT, dtype=np.float32)
        if len(x):
            m, s = self.topo_ensemble(name, x)
            mean[evaluable] = m
            std[evaluable] = s
        events = set_ak_column(events, f"topo_eff_{name}", mean)
        events = set_ak_column(events, f"topo_eff_{name}_std", std)

        # A CONDITIONAL estimator is P(target | conditioning bit fired, x). On its own the eps
        # column is NOT an efficiency of the target: it is only defined where the bit fired, and a
        # consumer that averages it over all events overstates the target. The usable quantity is
        # the product with the stored bit, which this writes explicitly so that no downstream
        # arithmetic has to remember the contract -- see TOPO_CONDITIONAL_NOTE.
        cond = self.topo_conditions.get(name)
        if cond is not None:
            if cond not in events.HLT.fields:
                # the reduction predates commit 39eea63 and does not carry this bit. There is no
                # safe default for a missing conditioning decision -- assuming it fired inflates
                # the weight, assuming it did not zeroes it -- so write EMPTY_FLOAT, which every
                # consumer already has to mask, and say so once per chunk.
                if not self._topo_warned_missing.get(cond):
                    logger.warning(
                        f"conditional estimator {name!r} needs the stored bit HLT.{cond}, which is "
                        f"not in these reduced columns; topo_weight_{name} is EMPTY_FLOAT "
                        f"everywhere. Re-reduce with the current keep_columns to use it",
                    )
                    self._topo_warned_missing[cond] = True
                w_cond = np.full(len(in_support), EMPTY_FLOAT, dtype=np.float64)
            else:
                bit = np.asarray(ak.to_numpy(events.HLT[cond]), dtype=np.float64)
                w_cond = np.where(
                    (mean != EMPTY_FLOAT) & in_support,
                    np.clip(mean.astype(np.float64), 0.0, 1.0) * bit,
                    EMPTY_FLOAT,
                )
            events = set_ak_column(events, f"topo_weight_{name}", w_cond.astype(np.float32))

    d_or2 = np.asarray(ak.to_numpy(events.topo_or2), dtype=np.float64)
    eff_res = np.asarray(ak.to_numpy(events[f"topo_eff_{TOPO_RESIDUAL_ESTIMATOR}"]), dtype=np.float64)
    eff_or2 = np.asarray(ak.to_numpy(events.topo_eff_or2), dtype=np.float64)

    # The residual estimator is a JOINT probability, eps_res(x) = P(TOPO & ~OR2 | x) -- its target
    # is ``topo & ~or2`` over every preselected event, not over the OR2-failing ones. So it already
    # carries the ~OR2 requirement, and multiplying it by the stored (1 - d_OR2) applies that
    # requirement a SECOND time. What is wanted on an OR2-failing event is the conditional
    # P(TOPO | ~OR2, x) = P(TOPO & ~OR2 | x) / P(~OR2 | x), and the emulated OR2 efficiency is
    # exactly the denominator.
    #
    # The correction is not small. Measured on reduced 2024 signal, the double-counted form gives a
    # TOPO gain of 2.3 pp where the conditional form gives 8.0 pp, because E[(1 - d_OR2) eps_res] is
    # smaller than E[eps_res] by a factor of P(~OR2) ~ 0.3.
    #
    # In expectation the conditional form is exact: E[(1 - d_OR2) | x] = 1 - eps_OR2(x), which
    # cancels the denominator and leaves E[w] = P(OR2) + E[eps_res] = P(OR3). The one assumption is
    # that eps_OR2 estimates P(OR2 | x) well -- and that is not an assumption we have to take on
    # faith, because it is precisely what hbw.TopoEmulationClosure measures against the stored bit.
    #
    # The in_support gate below is explicit as of v4. It is not a change of behaviour: until v3
    # eff_res was 0.0 outside the support, so eff_cond came out 0 and w_built collapsed to d_or2
    # there anyway. Now that the estimators extrapolate onto real features outside the support,
    # that accident no longer holds, and the gate has to be written down to keep both weights
    # bit-for-bit what they were. Extending the WEIGHT beyond the support is a separate physics
    # decision and is deliberately not taken here.
    p_fail = np.clip(1.0 - eff_or2, 1e-6, None)
    eff_cond = np.where(in_support, np.clip(eff_res / p_fail, 0.0, 1.0), 0.0)
    w_built = np.clip(d_or2 + (1.0 - d_or2) * eff_cond, 0.0, 1.0)
    events = set_ak_column(events, "topo_trigger_weight_built", w_built.astype(np.float32))

    # the applied weight: one directly fitted efficiency, so no SF_OR2 rides on a term worth 86%
    # of the weight and the b-tagged Mu12 leg never reaches a denominator. Outside the support
    # there is no model, and the stored OR2 decision is the floor.
    eff_w = np.asarray(ak.to_numpy(events[f"topo_eff_{self.weight_estimator}"]), dtype=np.float64)
    w = np.where(in_support, np.clip(eff_w, 0.0, 1.0), d_or2)   # unchanged in v4
    events = set_ak_column(events, "topo_trigger_weight", w.astype(np.float32))

    # Internal consistency of estimators that were fitted independently of one another: nothing in
    # the fit enforces OR3 >= OR2 or OR3 >= TOPO, yet both hold by construction of the targets. The
    # violation rate is therefore a direct, assumption-free measure of how far the six estimators
    # are from being mutually coherent, and it costs one comparison.
    if in_support.any():
        eff_topo = np.asarray(ak.to_numpy(events.topo_eff_topo), dtype=np.float64)[in_support]
        eff_or3 = np.asarray(ak.to_numpy(events.topo_eff_or3), dtype=np.float64)[in_support]
        n_sup = int(in_support.sum())
        bad_or2 = int((eff_or3 < eff_or2[in_support]).sum())
        bad_topo = int((eff_or3 < eff_topo).sum())
        if bad_or2 or bad_topo:
            logger.info(
                f"estimator ordering violated on {bad_or2} ({100 * bad_or2 / n_sup:.2f}%) events "
                f"for eps_OR3 >= eps_OR2 and {bad_topo} ({100 * bad_topo / n_sup:.2f}%) for "
                f"eps_OR3 >= eps_TOPO; both are exact for the targets, so this is emulation noise",
            )

    return events


@topo_or3_weights.requires
def topo_or3_weights_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@topo_or3_weights.setup
def topo_or3_weights_setup(self: Producer, reqs: dict, **kwargs) -> None:
    """
    Load the exported ensemble bundle and build one xgboost Booster per member.

    The evaluation below reproduces ``nd_eff._train_one``'s closure BIT-FOR-BIT, which was
    measured, not assumed. Three details each worth ~1e-7 if got wrong:

    1. the ``eps`` clip is applied to the RAW booster probability, BEFORE the logit
       (``nd_eff.py:34-36``);
    2. the logit stays in **float32** -- xgboost predicts float32 and the reference implementation
       hands that array straight to sklearn; computing it in float64 moves the result by 2.2e-7;
    3. the Platt fold is then done in **float64** with ``scipy.special.expit``. Under NumPy 2's
       weak-scalar promotion ``a * z32 + b`` with Python floats stays float32 (worth 7.0e-8), so
       the cast must be explicit, and ``1/(1+exp(-x))`` differs from ``expit`` by one ulp because
       ``LogisticRegression.predict_proba`` uses the latter.
    """
    import gzip
    import json

    import xgboost as xgb
    from scipy.special import expit

    path = reqs["external_files"].files.topo_ensemble.abspath
    with gzip.open(path, "rt", encoding="utf-8") as f:
        bundle = json.load(f)

    # the bundle ships the feature order it was fitted on; disagreeing with it is a hard failure,
    # because a silently reordered feature vector produces plausible numbers and no symptom.
    shipped = tuple(bundle["feature_order"])
    if shipped != FEATURE_ORDER:
        raise ValueError(
            f"topo ensemble was fitted on features {shipped} but this producer builds "
            f"{FEATURE_ORDER}; re-export the model or fix FEATURE_ORDER",
        )
    missing = set(TOPO_ESTIMATORS) - set(bundle["estimators"])
    if missing:
        raise ValueError(f"topo ensemble bundle {path} is missing estimators {sorted(missing)}")

    # A conditional estimator applied as if it were unconditional is silent and wrong (see
    # TOPO_CONDITIONAL_NOTE), so the weight may only ever be an UNCONDITIONAL estimator. Guard it
    # here rather than at use time: by then the number is plausible and nothing distinguishes it.
    w_spec = bundle["estimators"].get(self.weight_estimator, {})
    if w_spec.get("conditioned_on") is not None:
        raise ValueError(
            f"weight_estimator {self.weight_estimator!r} is conditioned on "
            f"{w_spec['conditioned_on']!r}; a conditional efficiency is P(target | that bit fired) "
            f"and must be applied as eps * stored_bit (column topo_weight_*), never as a weight on "
            f"its own",
        )

    if self.weight_estimator not in TOPO_WEIGHT_CHOICES:
        raise ValueError(
            f"weight_estimator {self.weight_estimator!r} is not one of {TOPO_WEIGHT_CHOICES}; "
            f"note that {TOPO_RESIDUAL_ESTIMATOR!r} is a residual and not an efficiency, so it "
            f"cannot be a weight on its own",
        )

    self.topo_eps = float(bundle["eps"])
    self.topo_meta = bundle["provenance"]
    self.topo_members = {}
    self.topo_conditions = {}
    self._topo_warned_missing = {}
    # the v1 bundle has six estimators, v2 fifteen; take whatever is there so both load.
    for name in TOPO_ESTIMATORS + tuple(n for n in TOPO_EXTRA_ESTIMATORS if n in bundle["estimators"]):
        spec = bundle["estimators"][name]
        # ``conditioned_on`` must be present on EVERY estimator in a v2-contract bundle (explicit
        # null for the unconditional ones). A bundle that omits the key predates the contract and
        # its conditionals would be indistinguishable from unconditional ones, so refuse it.
        if "_given_" in name and "conditioned_on" not in spec:
            raise ValueError(
                f"estimator {name!r} in {path} has no 'conditioned_on' key; this bundle predates "
                f"the conditional contract and cannot be applied safely -- re-export it",
            )
        cond = spec.get("conditioned_on")
        if cond is not None:
            col = cond[len("HLT_"):] if cond.startswith("HLT_") else cond
            self.topo_conditions[name] = col
        self.topo_members[name] = (
            list(map(int, spec["features"])),
            [
                (
                    xgb.Booster(model_file=bytearray(m["booster"], "utf-8")),
                    float(m["platt_a"]),
                    float(m["platt_b"]),
                )
                for m in spec["members"]
            ],
        )

    logger.info(
        f"loaded TOPO ensemble from {path}: "
        f"{', '.join(f'{n}(K={len(v[1])})' for n, v in self.topo_members.items())}; "
        f"provenance {self.topo_meta}",
    )

    eps = self.topo_eps

    def calibrate(booster, a, b, dm):
        """One member: the booster probability, clipped, logit-ed, Platt-folded."""
        p = booster.predict(dm)                       # float32, as the reference path has it
        p = np.clip(p, eps, 1.0 - eps)                # clip the RAW probability, before the logit
        z = np.log(p / (1.0 - p)).astype(np.float64)  # float32 logit, then explicit widening
        return expit(a * z + b)

    def topo_ensemble(name: str, x: np.ndarray):
        """(mean, std) over the K calibrated members, on the rows given."""
        features, members = self.topo_members[name]
        dm = xgb.DMatrix(np.ascontiguousarray(x[:, features]))
        preds = np.column_stack([calibrate(booster, a, b, dm) for booster, a, b in members])
        return preds.mean(1), preds.std(1)

    self.topo_ensemble = topo_ensemble


@topo_or3_weights.teardown
def topo_or3_weights_teardown(self: Producer, **kwargs) -> None:
    for attr in ("topo_members", "topo_ensemble", "topo_eps", "topo_meta", "topo_conditions",
                 "_topo_warned_missing"):
        if hasattr(self, attr):
            delattr(self, attr)
