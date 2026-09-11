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

#: feature column order. Load-bearing: it is fixed by ``config.py``'s ``ALL_FEATURES`` and must
#: match the order the estimator was fit on. Never reorder without re-exporting the model.
FEATURE_ORDER = ("mu_pt", "mu_eta", "mu_iso", "ht", "lead_btag", "lead_bpt")


@producer(
    uses={
        "Jet.{pt,eta,phi,mass,btagPNetB}",
        "Muon.{pt,eta,phi,mass,pfRelIso03_all,tightId}",
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

#: the estimator whose output enters the OR3 weight.
TOPO_RESIDUAL_ESTIMATOR = "topo_res"


@producer(
    uses={topo_or2_weights, topo_isomu24_weights} | {f"topo_feat.{f}" for f in FEATURE_ORDER} | {
        "topo_feat.valid", "topo_feat.n_btag_pnet",
    },
    produces={topo_or2_weights, topo_isomu24_weights, "topo_trigger_weight", "topo_in_support"} | {
        f"topo_eff_{n}" for n in TOPO_ESTIMATORS
    } | {
        f"topo_eff_{n}_std" for n in TOPO_ESTIMATORS
    },
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_topo.sh"),
    mc_only=True,
    #: minimum number of PNet b-tags. NOT cosmetic: the estimator's training preselection is
    #: ``n_mu>=1 & n_jet>=3 & n_btag_pnet>=2 & n_ele_tight==0`` while ``topo_feat.valid`` is only
    #: the first two terms, so ``valid`` alone would extrapolate the model outside its support.
    min_n_btag_pnet=2,
    version=2,
)
def topo_or3_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    The OR3 arm: the stored OR2 decision plus the emulated TOPO residual.

    .. code-block:: text

        w_OR3 = d_OR2 * SF_OR2  +  (1 - d_OR2) * eps_res(c) / (1 - eps_OR2(c))

    with ``SF_OR2 == 1`` for now (see :py:func:`topo_or2_weights`) and ``eps_res`` the calibrated
    ensemble mean for the residual target ``TOPO & ~OR2``. The division is not cosmetic -- see the
    comment at the weight itself; ``eps_res`` is a joint probability, so without it the ``~OR2``
    condition is applied twice and the TOPO gain comes out several times too small.

    Two choices that look like details and are not:

    * **The residual target, not the marginal.** Fitting ``TOPO`` alone and combining by
      inclusion-exclusion would assume leg factorisation, which the trigger-efficiency study
      measures and finds does not hold -- TOPO is heavily correlated with the muon leg. Fitting
      ``TOPO & ~OR2`` assumes nothing.
    * **A probability weight, not a decision.** ``w = eps``, not ``w = 1{eps > 0.5}``. A threshold
      is simply biased (``E[1{eps>0.5}] != E[eps]``) and a Bernoulli draw throws away precision for
      nothing; the probability weight is unbiased in yield and correct in shape.

    Outside the training support the weight falls back to the stored ``d_OR2``, i.e. the TOPO leg
    is credited with nothing rather than with an extrapolation.

    Besides the weight this writes ``topo_eff_<name>`` and ``topo_eff_<name>_std`` for every
    estimator in the bundle. Three of the four have a stored decision in Summer24 NanoAODv15 and
    are emulated *only* so the emulation can be checked against it: ``topo_eff_isomu24`` pairs with
    ``topo_isomu24``, ``topo_eff_or2`` with ``topo_or2``, and ``topo_eff_mu12`` with the stored HLT
    bit. Only ``topo_res`` is load-bearing, TOPO being the one path absent from the 2024 menu.
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

    # the feature matrix is built for in-support rows ONLY. Elsewhere the columns carry the
    # EMPTY_FLOAT sentinel, which is finite and would therefore NOT be treated as missing by
    # XGBoost -- it would be read as a real, very negative feature value.
    x = np.column_stack([
        np.asarray(ak.to_numpy(feat[f]), dtype=np.float64)[in_support] for f in FEATURE_ORDER
    ]) if in_support.any() else np.zeros((0, len(FEATURE_ORDER)))

    n_out = int((~in_support).sum())
    if n_out:
        logger.info(
            f"{n_out} of {len(in_support)} events ({n_out / len(in_support) * 100:.2f}%) are "
            f"outside the estimator's training support; falling back to w = d_OR2 there",
        )

    for name in TOPO_ESTIMATORS:
        mean = np.zeros(len(in_support), dtype=np.float32)
        std = np.zeros(len(in_support), dtype=np.float32)
        if len(x):
            m, s = self.topo_ensemble(name, x)
            mean[in_support] = m
            std[in_support] = s
        events = set_ak_column(events, f"topo_eff_{name}", mean)
        events = set_ak_column(events, f"topo_eff_{name}_std", std)

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
    p_fail = np.clip(1.0 - eff_or2, 1e-6, None)
    eff_cond = np.clip(eff_res / p_fail, 0.0, 1.0)
    w = np.clip(d_or2 + (1.0 - d_or2) * eff_cond, 0.0, 1.0)
    events = set_ak_column(events, "topo_trigger_weight", w.astype(np.float32))

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

    self.topo_eps = float(bundle["eps"])
    self.topo_meta = bundle["provenance"]
    self.topo_members = {}
    for name in TOPO_ESTIMATORS:
        spec = bundle["estimators"][name]
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
    for attr in ("topo_members", "topo_ensemble", "topo_eps", "topo_meta"):
        if hasattr(self, attr):
            delattr(self, attr)
