# coding: utf-8

"""
Hist producer for the closure test of the TOPO trigger emulation.

WHAT THIS MEASURES
For three of the six estimators shipped in the exported ensemble the target trigger is a *stored
bit* in Summer24 NanoAODv15, so the emulated efficiency can be compared to truth on our own events:

======================  ==========================================  ===============================
estimator               emulated column                             stored truth
======================  ==========================================  ===============================
``isomu24``             ``topo_eff_isomu24``                        ``topo_isomu24``
``mu12``                ``topo_eff_mu12``                           ``HLT.Mu12_IsoVVL_PFHT150_PNetBTag0p53``
``or2``                 ``topo_eff_or2``                            ``topo_or2``
======================  ==========================================  ===============================

The other three -- ``topo_res``, ``topo`` and ``or3``, the residual, the marginal TOPO path and the
directly fitted three-path union -- have no truth anywhere in 2024, TOPO not being in the menu,
which is the whole reason any of this is emulated. They are carried as *predictions only* and never
enter a residual; their plotted bands are the ensemble spread, not a closure. ``or3`` is the one the
applied weight is taken from, so it is the one the analysis actually stands on.

HOW ONE PASS PRODUCES ALL OF IT
Everything is filled into a single histogram carrying one extra ``arm`` axis, the same trick
``hbw.tasks.trigger_sf.ComputeTriggerSF`` uses with its ``trig_ids`` axis: efficiencies are then
ratios of slices of one object rather than ratios across separate task outputs, so numerator and
denominator can never drift apart in binning, selection or merge state.

The arm weights are, per event, with ``s = topo_in_support``:

* ``den_all`` -- 1, every event, so the in-support *fraction* is readable per bin
* ``den``     -- ``s``, the denominator of every efficiency below
* ``dir_<e>`` -- ``y_e * s``   (stored decision)
* ``emu_<e>`` -- ``f_e * s``   (ensemble mean)
* ``dif_<e>`` -- ``(f_e - y_e) * s``
* ``std_<e>`` -- ``sigma_e * s`` (ensemble spread, i.e. the model's own uncertainty)
* ``emu_or3_built`` -- the OR2 + residual construction, kept as a cross-check on ``emu_or3``
* ``gain_or3`` / ``gain_emu`` -- the applied weight minus the stored / the emulated OR2

WHY ``dif_`` IS A SEPARATE ARM AND NOT A SUBTRACTION AFTERWARDS
``f`` and ``y`` are measured on the *same events*, so they are strongly correlated and the error on
their difference is NOT the quadrature sum of two efficiency errors -- that would overestimate it by
roughly ``1/sqrt(1-rho)`` and quietly destroy the test's power. The correct error is the paired one,
``std(f - y) / sqrt(N)``, and ``hist``'s ``Weight`` storage hands it over for free: filling with the
per-event difference gives ``sum(d)`` in ``values()`` and ``sum(d**2)`` in ``variances()``, which is
all the paired variance needs.

That identity holds only while the event weight is 1, which is why the unweighted producer is the
default and the weighted twin exists for the yield statement alone.

ON EMPTY_FLOAT
Undefined variable values are written as ``EMPTY_FLOAT = -99999``, which is FINITE and therefore
lands in a real underflow bin rather than being dropped. Per-bin efficiencies are unaffected (both
arms of the ratio land in the same bin), but any *integral* over this histogram must be taken with
``flow=True`` or the undefined events silently vanish from both numerator and denominator.
"""

from __future__ import annotations

import law
import order as od

from columnflow.histogramming import HistProducer
from columnflow.histogramming.default import cf_default
from columnflow.hist_util import (
    create_hist_from_variables, fill_hist, translate_hist_intcat_to_strcat,
)
from columnflow.columnar_util import Route
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Any

np = maybe_import("numpy")
ak = maybe_import("awkward")
if TYPE_CHECKING:
    hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


#: estimator name -> route of the stored decision it is supposed to reproduce.
#: ``mu12`` reads the raw HLT branch because, unlike the other two, no producer writes it out as a
#: dedicated column; it survives ReduceEvents, so this costs nothing.
TOPO_CLOSURE_TARGETS = {
    "isomu24": "topo_isomu24",
    "mu12": "HLT.Mu12_IsoVVL_PFHT150_PNetBTag0p53",
    "or2": "topo_or2",
}

#: estimators with no truth in the 2024 menu: predicted, never closed.
#:
#: ``topo_res`` is the residual that builds the weight, ``topo`` the marginal TOPO efficiency, and
#: ``or3`` the directly fitted three-path union. None of the three can be closed here -- TOPO is
#: absent from the 2024 menu, which is the entire reason any of this is emulated -- so all three
#: are reported with their ensemble spread and never with a residual.
TOPO_PREDICTION_ONLY = ("topo_res", "topo", "or3")

#: the arm axis, in fill order. Kept as a module constant so the reading task cannot disagree with
#: the writing producer about what a slice means.
TOPO_CLOSURE_ARMS = tuple(
    ["den_all", "den"] +
    [f"{p}_{e}" for e in TOPO_CLOSURE_TARGETS for p in ("dir", "emu", "dif", "std")] +
    [f"{p}_{e}" for e in TOPO_PREDICTION_ONLY for p in ("emu", "std")] +
    # ``emu_or3_built`` is the OR2 + residual construction, which is no longer the applied weight
    # but is kept because it estimates the same quantity as the directly fitted ``emu_or3`` by an
    # independent route -- their difference is the sharpest statement the closure can make about
    # the emulation, and nothing else in the chain makes it.
    #
    # The two gains are the applied weight against the stored OR2 decision and against the
    # *emulated* one. The second is the apples-to-apples number: both sides then come from the same
    # fit on the same events, so the part of each estimator's bias that is common to them cancels,
    # whereas model-minus-truth carries the OR3 closure bias into the gain at full size.
    #
    # Both are carried as their own arms rather than subtracted afterwards for the same reason the
    # dif_ arms are: they are per-event differences on the same events, so the Weight storage hands
    # over the paired error, and the quadrature sum of two efficiency errors would be wrong.
    ["emu_or3_built", "gain_or3", "gain_emu"],
)


@cf_default.hist_producer(
    uses=(
        {"topo_in_support", "topo_trigger_weight", "topo_trigger_weight_built"} |
        set(TOPO_CLOSURE_TARGETS.values()) |
        {f"topo_eff_{e}" for e in list(TOPO_CLOSURE_TARGETS) + list(TOPO_PREDICTION_ONLY)} |
        {f"topo_eff_{e}_std" for e in list(TOPO_CLOSURE_TARGETS) + list(TOPO_PREDICTION_ONLY)}
    ),
    mc_only=True,
    #: event weight applied on top of every arm factor. ``None`` means 1, which is what makes the
    #: paired-error identity above exact. Only override it for a yield-level statement.
    weight_column=None,
    version=2,
)
def topo_closure(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    if self.weight_column is None:
        return events, ak.Array(np.ones(len(events), dtype=np.float64))
    return events, ak.values_astype(Route(self.weight_column).apply(events), np.float64)


@topo_closure.create_hist
def topo_closure_create_hist(
    self: HistProducer,
    variables: list[od.Variable],
    task: law.Task,
) -> hist.Hist:
    return create_hist_from_variables(
        *variables,
        categorical_axes=[
            ("category", "intcat"),
            ("process", "intcat"),
            ("shift", "intcat", [0]),
            # an INT axis translated to strings in post-processing, exactly as columnflow does for
            # category/process/shift. A StrCategory cannot be filled here: ``category_ids`` is
            # jagged, so ``fill_hist`` takes its cartesian-product path, and that path rejects
            # string data outright.
            ("arm", "intcat", list(range(len(TOPO_CLOSURE_ARMS)))),
        ],
        weight=True,
    )


@topo_closure.fill_hist
def topo_closure_fill_hist(
    self: HistProducer,
    h: hist.Hist,
    data: dict[str, Any],
    variables: list[od.Variable],
    events: ak.Array,
    task: law.Task,
) -> None:
    def col(route: str) -> np.ndarray:
        return ak.to_numpy(Route(route).apply(events)).astype(np.float64)

    base = ak.to_numpy(data["weight"]).astype(np.float64)
    support = col("topo_in_support")

    factors = {"den_all": np.ones_like(support), "den": support}
    for est, truth_route in TOPO_CLOSURE_TARGETS.items():
        y = col(truth_route)
        f = col(f"topo_eff_{est}")
        factors[f"dir_{est}"] = y * support
        factors[f"emu_{est}"] = f * support
        factors[f"dif_{est}"] = (f - y) * support
        factors[f"std_{est}"] = col(f"topo_eff_{est}_std") * support
    for est in TOPO_PREDICTION_ONLY:
        factors[f"emu_{est}"] = col(f"topo_eff_{est}") * support
        factors[f"std_{est}"] = col(f"topo_eff_{est}_std") * support
    w = col("topo_trigger_weight")
    factors["emu_or3_built"] = col("topo_trigger_weight_built") * support
    factors["gain_or3"] = (w - col("topo_or2")) * support
    factors["gain_emu"] = (w - col("topo_eff_or2")) * support

    for arm, factor in factors.items():
        fill_hist(
            h,
            {
                **data,
                "arm": np.full(len(base), TOPO_CLOSURE_ARMS.index(arm), dtype=np.int32),
                "weight": base * factor,
            },
            last_edge_inclusive=task.last_edge_inclusive,
        )


@topo_closure.post_process_hist
def topo_closure_post_process_hist(self: HistProducer, h: hist.Hist, task: law.Task) -> hist.Hist:
    """
    Translate the integer axes to strings, ``arm`` included.

    This repeats columnflow's default post-processing rather than delegating to it: the default hook
    is registered on ``cf_default`` and overriding it here replaces it wholesale, so the three
    standard translations have to be carried along or downstream tasks lose their string lookups.
    """
    axis_names = {ax.name for ax in h.axes}
    if "arm" in axis_names:
        h = translate_hist_intcat_to_strcat(h, "arm", lambda i: TOPO_CLOSURE_ARMS[i])
    if "category" in axis_names:
        h = translate_hist_intcat_to_strcat(h, "category", lambda i: self.config_inst.get_category(i).name)
    if "process" in axis_names:
        h = translate_hist_intcat_to_strcat(h, "process", lambda i: self.config_inst.get_process(i).name)
    if "shift" in axis_names:
        h = translate_hist_intcat_to_strcat(h, "shift", lambda i: self.config_inst.get_shift(i).name)
    return h


#: yield-level twin. The per-bin efficiencies stay meaningful (both arms carry the same weight), but
#: the paired error is NOT valid here -- see the module docstring -- so the reading task refuses to
#: quote pulls unless the unweighted producer was used.
topo_closure_norm = topo_closure.derive("topo_closure_norm", cls_dict={
    "weight_column": "dataset_normalization_weight",
    "uses": topo_closure.uses | {"dataset_normalization_weight"},
})
