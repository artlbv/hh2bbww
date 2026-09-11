# coding: utf-8

"""
Closure test of the TOPO trigger emulation, per sample and per variable.

THE QUESTION
The OR3 weight is ``d_OR2 + (1 - d_OR2) * eps_res``, and ``eps_res`` comes from a BDT ensemble that
was fitted somewhere else, on a different campaign and a different HLT menu. Nothing in the analysis
can check ``eps_res`` directly: TOPO does not exist in the 2024 menu, which is why it is emulated at
all. What *can* be checked is the transport itself. Three of the four estimators in the same bundle,
built from the same features by the same code, target triggers whose decisions are stored bits here.
Their closure is the only handle on whether the ensemble survives the move into this campaign, and
the residual it leaves is the natural size of the emulation systematic.

So this task answers, per sample and per variable: *does the emulated efficiency reproduce the
stored one, and where does it fail?*

WHAT IS AND IS NOT A CLOSURE HERE
``isomu24``, ``mu12`` and ``or2`` are closures. ``topo_res`` is not -- it is plotted as a prediction
with its ensemble spread, and no residual is quoted for it. Reading a ``topo_res`` band as a
validation would be reading the model's opinion of itself.

WHY THE DENOMINATOR IS ``topo_in_support`` AND NOT ``topo_feat.valid``
``valid`` is only ``n_mu >= 1 & n_jet >= 3``; the estimator's training preselection additionally
required ``n_btag_pnet >= 2`` and no tight electron. On reduced 2024 signal the two differ by more
than a factor of one and a half, so a ``valid``-based denominator would be scoring the model on
events it was never fitted on and calling the resulting bias a closure failure. The producer already
evaluates only in support and writes ``topo_in_support``; this task uses exactly that.

The in-support fraction is reported per bin anyway (arm ``den_all`` over arm ``den``), because how
much phase space the model declines to cover is itself a result.

PAIRED ERRORS, NOT QUADRATURE
See :py:mod:`hbw.weight.topo_closure`. The emulated and the stored efficiency are measured on the
same events; their difference has the paired error ``std(f - y) / sqrt(N)``, which is much smaller
than the quadrature sum. Quoting the latter would wash out exactly the biases this test exists to
find.

INTEGRALS TAKE THE FLOW
``EMPTY_FLOAT`` is finite and lands in underflow rather than being dropped, so every integral here
is taken with ``flow=True``. As a side effect the integrated efficiency must come out identical no
matter which variable it was integrated over; the task checks that and refuses to write if it does
not, which is a direct test that no flow bin was lost.
"""

from __future__ import annotations

import law
import luigi

from collections import defaultdict

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import maybe_import, dev_sandbox

from hbw.tasks.base import HBWTask
from hbw.tasks.histograms import HistogramsUserSingleShiftBase
from hbw.config.topo_an_variables import TOPO_AN_VARIABLES
from hbw.weight.topo_closure import TOPO_CLOSURE_TARGETS, TOPO_PREDICTION_ONLY

np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


#: the estimator inputs plus their two companions. These are the variables the model actually sees,
#: so a failure that is localised in one of them points at the model; a failure that is flat in all
#: of them points at the plumbing.
TOPO_ESTIMATOR_VARIABLES = (
    "topo_mu_pt", "topo_mu_pt_uni", "topo_mu_eta", "topo_mu_iso",
    "topo_ht", "topo_lead_btag", "topo_lead_bpt",
    "topo_l1ht_proxy", "topo_n_btag_pnet",
)

#: the full default set: the estimator inputs plus the sixteen kinematic classifier inputs of the
#: S/B study. The latter are variables the model has NEVER seen, which is the point -- a model that
#: closes in its own inputs and drifts in m(bb) or dR(b,b) is telling us the emulation reshapes the
#: very distributions the analysis cuts on, and that would not show up in any of the nine above.
TOPO_CLOSURE_VARIABLES = TOPO_ESTIMATOR_VARIABLES + tuple(TOPO_AN_VARIABLES.split(","))

#: sample families, by dataset-name prefix. Deliberately prefix matching rather than the config's
#: process tree: the point is to group what was *run*, and the datasets parameter is the thing the
#: user controls.
TOPO_SAMPLE_GROUPS = {
    "signal": ("hh_ggf_", "hh_vbf_"),
    "tt": ("tt_",),
    "st": ("st_",),
    "w_lnu": ("w_lnu_",),
    "dy": ("dy_",),
    "qcd_mu": ("qcd_mu_",),
    "vv": ("ww_", "wz_", "zz_"),
}


def _group_of(dataset: str) -> str:
    for group, prefixes in TOPO_SAMPLE_GROUPS.items():
        if any(dataset.startswith(p) for p in prefixes):
            return group
    return "other"


class TopoEmulationClosure(
    HBWTask,
    HistogramsUserSingleShiftBase,
):
    """
    Compare emulated to stored trigger efficiency, per sample and per variable.
    """

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # multi-config machinery is inherited; every consumer here works on one config, taken as the
    # first, so the task is invoked with --configs and reads config_insts[0].
    #: minimum number of in-support events in a bin before its residual is quoted. Bins below this
    #: are still written to the JSON but are not plotted and never enter the outlier list -- a
    #: three-event bin produces spectacular pulls that mean nothing.
    min_bin_events = luigi.IntParameter(
        default=50,
        description="minimum in-support events per bin for a residual to be quoted; default: 50",
    )

    hist_producer = HistogramsUserSingleShiftBase.hist_producer.copy(
        default="topo_closure",
        description="hist producer carrying the closure arms; default: topo_closure",
        add_default_to_description=True,
    )
    producers = HistogramsUserSingleShiftBase.producers.copy(
        default=("topo_or3_weights",),
        add_default_to_description=True,
    )
    variables = HistogramsUserSingleShiftBase.variables.copy(
        default=TOPO_CLOSURE_VARIABLES,
        add_default_to_description=True,
    )
    categories = HistogramsUserSingleShiftBase.categories.copy(
        default=("incl",),
        add_default_to_description=True,
    )
    ml_models = HistogramsUserSingleShiftBase.ml_models.copy(
        default=(),
        add_default_to_description=True,
    )
    shift = HistogramsUserSingleShiftBase.shift.copy(
        default="nominal",
        add_default_to_description=True,
    )

    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    @property
    def config_inst(self):
        return self.config_insts[0]

    def output(self):
        return {
            "json": self.target("topo_closure.json"),
            "summary": self.target("topo_closure_summary.txt"),
            "plots": self.target("plots", dir=True),
        }

    # -- the arithmetic ------------------------------------------------------------------------

    def _select_categories(self, h):
        """
        Restrict the category axis to the requested categories, then sum it.

        This is not optional book-keeping. ``category_ids`` carries *every* category an event
        belongs to, so an event that is in ``incl``, ``sr`` and ``1mu`` is filled three times and
        summing the whole axis counts it three times. Efficiencies survive that -- numerator and
        denominator are inflated by the same factor -- but N does not, and N is what sets every
        error here, so the residuals would come out looking two or three times more significant
        than they are.

        Which categories actually reach the axis depends on the selector and on the producers that
        were run, and it varies per dataset (a dataset with no events in a category simply has no
        entry). So the requested names are expanded to leaves, intersected with what is present,
        and a name that survives as itself is kept as itself.
        """
        import hist

        axis_names = list(h.axes["category"])
        wanted = []
        for name in self.categories:
            cat = self.config_inst.get_category(name)
            leaves = [c.name for c in (cat.get_leaf_categories() or [cat])]
            present = [c for c in leaves if c in axis_names]
            if not present and name in axis_names:
                present = [name]
            wanted.extend(present)
        wanted = list(dict.fromkeys(wanted))
        if not wanted:
            raise ValueError(
                f"none of the requested categories {list(self.categories)} (nor their leaves) are "
                f"present on the histogram, which carries {axis_names}",
            )
        return h[{"category": [hist.loc(c) for c in wanted]}][{"category": sum}]

    def _arm(self, h, arm: str, flow: bool):
        """``(values, variances)`` of one arm, reduced over the remaining categorical axes."""
        import hist
        sub = h[{"arm": hist.loc(arm)}]
        for ax in ("process", "shift"):
            if ax in {a.name for a in sub.axes}:
                sub = sub[{ax: sum}]
        if flow:
            # one variable axis is left; sum it including flow
            return sub.values(flow=True).sum(), sub.variances(flow=True).sum()
        return sub.values(flow=True), sub.variances(flow=True)

    def _residuals(self, h, flow: bool) -> dict:
        """
        Per-bin (or integrated, if *flow*) closure numbers for every estimator.

        Everything is derived from four arms: ``den_all`` (all events), ``den`` (in support), the
        estimator's ``dir_``/``emu_`` and its ``dif_``. The paired variance of the difference is
        ``E[d^2] - E[d]^2`` read straight off the ``dif_`` arm, whose ``variances()`` is ``sum(d^2)``
        because ``hist``'s Weight storage accumulates the square of whatever it is filled with.
        """
        h = self._select_categories(h)
        n_all, _ = self._arm(h, "den_all", flow)
        n, n_var = self._arm(h, "den", flow)

        # the paired-error identity only holds for unit event weights, where sum(w) == sum(w^2)
        unit_weights = bool(np.allclose(np.asarray(n), np.asarray(n_var), rtol=1e-6, atol=1e-9))

        with np.errstate(divide="ignore", invalid="ignore"):
            out = {
                "n_all": np.asarray(n_all),
                "n": np.asarray(n),
                "in_support_frac": np.where(np.asarray(n_all) > 0, np.asarray(n) / np.asarray(n_all), np.nan),
                "unit_weights": unit_weights,
                "estimators": {},
            }
            safe_n = np.where(np.asarray(n) > 0, np.asarray(n), np.nan)

            for est in TOPO_CLOSURE_TARGETS:
                s_dir, _ = self._arm(h, f"dir_{est}", flow)
                s_emu, _ = self._arm(h, f"emu_{est}", flow)
                s_dif, q_dif = self._arm(h, f"dif_{est}", flow)
                s_std, _ = self._arm(h, f"std_{est}", flow)

                eff_dir = np.asarray(s_dir) / safe_n
                eff_emu = np.asarray(s_emu) / safe_n
                delta = np.asarray(s_dif) / safe_n
                var_d = np.asarray(q_dif) / safe_n - delta ** 2
                sigma = np.sqrt(np.clip(var_d, 0.0, None) / safe_n) if unit_weights else np.full_like(delta, np.nan)

                out["estimators"][est] = {
                    "eff_direct": eff_dir,
                    "eff_direct_err": np.sqrt(np.clip(eff_dir * (1 - eff_dir), 0.0, None) / safe_n),
                    "eff_emulated": eff_emu,
                    "delta": delta,
                    "delta_err": sigma,
                    "pull": delta / np.where(sigma > 0, sigma, np.nan),
                    "model_std": np.asarray(s_std) / safe_n,
                }
            for est in TOPO_PREDICTION_ONLY:
                s_emu, _ = self._arm(h, f"emu_{est}", flow)
                s_std, _ = self._arm(h, f"std_{est}", flow)
                out["estimators"][est] = {
                    "eff_direct": np.full_like(np.asarray(s_emu, dtype=float), np.nan),
                    "eff_direct_err": np.full_like(np.asarray(s_emu, dtype=float), np.nan),
                    "eff_emulated": np.asarray(s_emu) / safe_n,
                    "delta": np.full_like(np.asarray(s_emu, dtype=float), np.nan),
                    "delta_err": np.full_like(np.asarray(s_emu, dtype=float), np.nan),
                    "pull": np.full_like(np.asarray(s_emu, dtype=float), np.nan),
                    "model_std": np.asarray(s_std) / safe_n,
                }
            # the constructed OR3, d_OR2 + (1 - d_OR2) * eps_res, and the gain it buys over the
            # stored OR2 decision. The gain gets the same paired treatment as the residuals: it is
            # a per-event difference on the same events, so its error is std(g)/sqrt(N).
            s_or3, _ = self._arm(h, "emu_or3_built", flow)
            s_gain, q_gain = self._arm(h, "gain_or3", flow)
            gain = np.asarray(s_gain) / safe_n
            var_g = np.asarray(q_gain) / safe_n - gain ** 2
            out["or3_built"] = np.asarray(s_or3) / safe_n
            out["gain"] = gain
            out["gain_err"] = (
                np.sqrt(np.clip(var_g, 0.0, None) / safe_n) if unit_weights
                else np.full_like(gain, np.nan)
            )
        return out

    # -- the run -------------------------------------------------------------------------------

    def run(self):
        import hist  # noqa: F401

        config = self.config_inst.name
        inputs = self.input()
        datasets = sorted(inputs[config].keys())
        variables = list(self.variables)

        logger.info(f"closure over {len(datasets)} datasets and {len(variables)} variables")

        # per dataset: variable -> histogram
        hists = defaultdict(dict)
        for dataset in datasets:
            for variable in variables:
                hists[dataset][variable] = self.load_histogram(inputs, config, dataset, variable)

        # integrated numbers, computed once per (dataset, variable). They must not depend on the
        # variable; if they do, a flow bin was lost somewhere and every differential number below
        # is suspect, so this is a hard failure rather than a warning.
        integrated = {}
        for dataset in datasets:
            per_var = {v: self._residuals(hists[dataset][v], flow=True) for v in variables}
            ref = per_var[variables[0]]
            for v in variables[1:]:
                for est in TOPO_CLOSURE_TARGETS:
                    a = ref["estimators"][est]["delta"]
                    b = per_var[v]["estimators"][est]["delta"]
                    if not np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True):
                        raise ValueError(
                            f"integrated closure for {dataset}/{est} depends on the variable it was "
                            f"integrated over ({variables[0]}: {a}, {v}: {b}); this means flow bins "
                            "are being lost -- check that EMPTY_FLOAT entries are summed with flow=True",
                        )
            integrated[dataset] = ref

        # differential numbers
        differential = {
            dataset: {v: self._residuals(hists[dataset][v], flow=False) for v in variables}
            for dataset in datasets
        }

        # write the machine-readable output
        def jsonify(obj):
            if isinstance(obj, np.ndarray):
                return [None if not np.isfinite(x) else float(x) for x in np.atleast_1d(obj)]
            if isinstance(obj, (np.floating, np.integer)):
                return None if not np.isfinite(obj) else float(obj)
            if isinstance(obj, dict):
                return {k: jsonify(v) for k, v in obj.items()}
            return obj

        payload = {
            "config": config,
            "categories": list(self.categories),
            "hist_producer": self.hist_producer,
            "min_bin_events": self.min_bin_events,
            "datasets": datasets,
            "variables": variables,
            "integrated": jsonify(integrated),
            "differential": {
                d: {v: jsonify(differential[d][v]) for v in variables} for d in datasets
            },
            "bin_edges": {
                v: [float(e) for e in self.config_inst.get_variable(v).bin_edges]
                for v in variables
            },
        }
        self.output()["json"].dump(payload, formatter="json", indent=2)

        # human-readable summary and plots
        self.output()["summary"].dump(self._summary_text(integrated, differential, datasets, variables),
                                      formatter="text")
        self._make_plots(differential, integrated, datasets, variables)

    def _summary_text(self, integrated, differential, datasets, variables) -> str:
        lines = []
        lines.append(f"TOPO emulation closure -- config {self.config_inst.name}, "
                     f"categories {','.join(self.categories)}, hist producer {self.hist_producer}")
        lines.append("")
        lines.append("Integrated, in-support events only. delta = emulated - stored, in percentage")
        lines.append("points; err is the PAIRED error std(f-y)/sqrt(N), not a quadrature sum.")
        lines.append("")
        header = f"{'dataset':<40s} {'N_supp':>9s} {'supp%':>6s}"
        for est in TOPO_CLOSURE_TARGETS:
            header += f" | {est:>10s} dir%   emu%   delta   err   pull"
        lines.append(header)
        lines.append("-" * len(header))

        group_acc = defaultdict(list)
        for dataset in datasets:
            r = integrated[dataset]
            row = f"{dataset:<40s} {float(r['n']):>9.0f} {100 * float(r['in_support_frac']):>5.1f}%"
            for est in TOPO_CLOSURE_TARGETS:
                e = r["estimators"][est]
                row += (
                    f" | {'':>10s} {100 * float(e['eff_direct']):6.2f} {100 * float(e['eff_emulated']):6.2f}"
                    f" {100 * float(e['delta']):+7.3f} {100 * float(e['delta_err']):6.3f}"
                    f" {float(e['pull']):+6.1f}"
                )
            lines.append(row)
            group_acc[_group_of(dataset)].append(dataset)

        lines.append("")
        lines.append("Predicted efficiencies with no truth in the 2024 menu, and the gain OR3 buys")
        lines.append("over the stored OR2 decision. eps_res is the JOINT residual P(TOPO & ~OR2 | x)")
        lines.append("and eps_TOPO the marginal. OR3(built) is d_OR2 + (1-d_OR2)*eps_res/(1-eps_OR2),")
        lines.append("OR3(fit) the directly fitted union. The two estimate the same quantity by")
        lines.append("different routes, so built-fit is a statement about the weight construction,")
        lines.append("not a closure -- it should sit at the size of the or2 residual above, and a")
        lines.append("large value means the construction is wrong rather than the transport.")
        lines.append("")
        head2 = (f"{'dataset':<40s} {'OR2 stored':>10s} {'eps_res':>8s} {'eps_TOPO':>9s} "
                 f"{'OR3 built':>10s} {'OR3 fit':>8s} {'built-fit':>10s} {'gain':>8s} {'err':>7s}")
        lines.append(head2)
        lines.append("-" * len(head2))
        for dataset in datasets:
            r = integrated[dataset]
            est = r["estimators"]
            built = 100 * float(r["or3_built"])
            fit = 100 * float(est["or3"]["eff_emulated"])
            lines.append(
                f"{dataset:<40s} {100 * float(est['or2']['eff_direct']):>9.2f}% "
                f"{100 * float(est['topo_res']['eff_emulated']):>7.2f}% "
                f"{100 * float(est['topo']['eff_emulated']):>8.2f}% "
                f"{built:>9.2f}% {fit:>7.2f}% {built - fit:>+9.3f} "
                f"{100 * float(r['gain']):>+7.3f} {100 * float(r['gain_err']):>6.3f}",
            )

        lines.append("")
        lines.append("By sample family (event-weighted mean of the per-dataset residuals):")
        for group, members in sorted(group_acc.items()):
            ns = np.array([float(integrated[d]["n"]) for d in members])
            if ns.sum() <= 0:
                continue
            row = f"  {group:<12s} N={ns.sum():>10.0f}"
            for est in TOPO_CLOSURE_TARGETS:
                d = np.array([float(integrated[m]["estimators"][est]["delta"]) for m in members])
                s = np.array([float(integrated[m]["estimators"][est]["delta_err"]) for m in members])
                # a member with no in-support events carries NaN, and 0 * NaN is NaN, which would
                # poison the whole family. It contributes nothing, so drop it rather than weight it.
                d, s = np.nan_to_num(d), np.nan_to_num(s)
                mean = float(np.sum(ns * d) / ns.sum())
                err = float(np.sqrt(np.sum((ns * s) ** 2)) / ns.sum())
                row += f" | {est}: {100 * mean:+7.3f} +- {100 * err:6.3f} pp"
            g = np.nan_to_num(np.array([float(integrated[m]["gain"]) for m in members]))
            ge = np.nan_to_num(np.array([float(integrated[m]["gain_err"]) for m in members]))
            row += (f" | OR3 gain: {100 * float(np.sum(ns * g) / ns.sum()):+7.3f} +- "
                    f"{100 * float(np.sqrt(np.sum((ns * ge) ** 2)) / ns.sum()):6.3f} pp")
            lines.append(row)

        # outliers: bins whose residual is significant and which hold enough events to be believed
        lines.append("")
        lines.append(f"Bins with |pull| > 3 and N >= {self.min_bin_events}:")
        found = 0
        for dataset in datasets:
            for variable in variables:
                edges = self.config_inst.get_variable(variable).bin_edges
                r = differential[dataset][variable]
                n = np.asarray(r["n"])
                for est in TOPO_CLOSURE_TARGETS:
                    e = r["estimators"][est]
                    pull = np.asarray(e["pull"])
                    bad = np.where((np.abs(pull) > 3) & (n >= self.min_bin_events))[0]
                    for i in bad:
                        # index 0 is underflow; the variable axis has len(edges) - 1 real bins
                        lo = "-inf" if i == 0 else f"{edges[i - 1]:g}"
                        hi = "+inf" if i >= len(edges) else f"{edges[i]:g}"
                        lines.append(
                            f"  {dataset:<38s} {variable:<20s} {est:<9s} [{lo},{hi})"
                            f" N={n[i]:>8.0f} delta={100 * float(e['delta'][i]):+7.3f} pp"
                            f" pull={float(pull[i]):+6.1f}",
                        )
                        found += 1
        if not found:
            lines.append("  none")
        lines.append("")
        return "\n".join(lines)

    def _make_plots(self, differential, integrated, datasets, variables) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        out = self.output()["plots"]
        out.touch()

        groups = defaultdict(list)
        for d in datasets:
            groups[_group_of(d)].append(d)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        #: quantities that are averages over events and therefore combine across datasets as an
        #: N-weighted mean. ``delta_err`` is not among them: it is an error and adds in quadrature.
        mean_keys = ("eff_direct", "eff_direct_err", "eff_emulated", "delta", "model_std")

        #: the panels of each page. "closure" has truth and shows a residual; "predict" has none
        #: and shows the ensemble spread; "gain" is the constructed OR3 against the stored OR2,
        #: with the acceptance it buys underneath.
        panels = (
            [(e, "closure") for e in TOPO_CLOSURE_TARGETS] +
            [(e, "predict") for e in TOPO_PREDICTION_ONLY] +
            [("or3_built", "gain")]
        )

        def combine(members, variable):
            """N-weighted mean of the per-dataset quantities over one sample family."""
            n_tot, acc, var = None, defaultdict(lambda: defaultdict(float)), defaultdict(float)
            extra = defaultdict(float)
            for m in members:
                r = differential[m][variable]
                n = np.asarray(r["n"], dtype=float)
                n_tot = n if n_tot is None else n_tot + n
                for est, e in r["estimators"].items():
                    for key in mean_keys:
                        acc[est][key] = acc[est][key] + np.nan_to_num(np.asarray(e[key], dtype=float)) * n
                    var[est] = var[est] + np.nan_to_num(np.asarray(e["delta_err"], dtype=float)) ** 2 * n ** 2
                for key in ("or3_built", "gain"):
                    extra[key] = extra[key] + np.nan_to_num(np.asarray(r[key], dtype=float)) * n
                extra["gain_var"] = extra["gain_var"] + np.nan_to_num(
                    np.asarray(r["gain_err"], dtype=float),
                ) ** 2 * n ** 2
            safe = np.where(n_tot > 0, n_tot, np.nan)
            res = {"n": n_tot, "estimators": {}}
            for est in acc:
                res["estimators"][est] = {k: acc[est][k] / safe for k in mean_keys}
                res["estimators"][est]["delta_err"] = np.sqrt(var[est]) / safe
            res["or3_built"] = extra["or3_built"] / safe
            res["gain"] = extra["gain"] / safe
            res["gain_err"] = np.sqrt(extra["gain_var"]) / safe
            return res

        pdf_path = out.child("topo_closure.pdf", type="f").abspath
        with PdfPages(pdf_path) as pdf:
            for variable in variables:
                var_inst = self.config_inst.get_variable(variable)
                edges = np.asarray(var_inst.bin_edges, dtype=float)
                centres = 0.5 * (edges[:-1] + edges[1:])
                # values(flow=True) is [underflow, bins..., overflow]; drop both ends for the plot,
                # they are in the JSON and in the integrated numbers
                inner = slice(1, len(edges))

                fig, axes = plt.subplots(
                    2, len(panels), figsize=(3.3 * len(panels), 5.6), sharex="col",
                    gridspec_kw={"height_ratios": [2, 1]},
                )
                for j, (est, kind) in enumerate(panels):
                    ax, rax = axes[0][j], axes[1][j]
                    for k, (group, members) in enumerate(sorted(groups.items())):
                        c = colors[k % len(colors)]
                        r = combine(members, variable)
                        mask = r["n"][inner] >= self.min_bin_events
                        if not mask.any():
                            continue

                        if kind == "gain":
                            # the constructed OR3 against the stored OR2 it is built on, with the
                            # acceptance it buys underneath
                            stored = r["estimators"]["or2"]["eff_direct"][inner]
                            built = r["or3_built"][inner]
                            ax.stairs(np.where(mask, built, np.nan), edges, baseline=None,
                                      color=c, ls="--", label=f"{group} (OR3 built)")
                            ax.errorbar(
                                centres[mask], stored[mask],
                                yerr=r["estimators"]["or2"]["eff_direct_err"][inner][mask],
                                fmt="o", ms=3, color=c, label=f"{group} (OR2 stored)",
                            )
                            rax.errorbar(
                                centres[mask], 100 * r["gain"][inner][mask],
                                yerr=100 * r["gain_err"][inner][mask], fmt="o", ms=3, color=c,
                            )
                            continue

                        e = r["estimators"][est]
                        emu = e["eff_emulated"][inner]
                        spread = e["model_std"][inner]
                        # the emulated efficiency is a per-bin average, so draw it as a step over
                        # the bin it belongs to rather than as a line through bin centres
                        # baseline=None keeps stairs from closing down to zero at the ends, which
                        # would draw a vertical line that reads as a real drop in efficiency
                        ax.stairs(np.where(mask, emu, np.nan), edges, baseline=None, color=c,
                                  ls="--", label=f"{group} (emulated)")
                        ax.stairs(np.where(mask, emu + spread, np.nan), edges,
                                  baseline=np.where(mask, emu - spread, np.nan),
                                  fill=True, alpha=0.15, color=c)
                        if kind == "closure":
                            ax.errorbar(
                                centres[mask], e["eff_direct"][inner][mask],
                                yerr=e["eff_direct_err"][inner][mask],
                                fmt="o", ms=3, color=c, label=f"{group} (stored)",
                            )
                            rax.errorbar(
                                centres[mask], 100 * e["delta"][inner][mask],
                                yerr=100 * e["delta_err"][inner][mask],
                                fmt="o", ms=3, color=c,
                            )
                        else:
                            # no truth exists for this one; the band is the model's own spread
                            rax.errorbar(
                                centres[mask], 100 * spread[mask], yerr=None,
                                fmt="o", ms=3, color=c,
                            )
                    title = {
                        "closure": est,
                        "predict": f"{est}  (no truth: prediction)",
                        "gain": "OR3 built vs OR2 stored",
                    }[kind]
                    ax.set_title(title, fontsize=7)
                    ax.set_ylim(-0.05, 1.25)
                    ax.set_ylabel("efficiency")
                    rax.axhline(0.0, color="k", lw=0.8)
                    rax.set_ylabel({
                        "closure": "emu - stored [pp]",
                        "predict": "ensemble spread [pp]",
                        "gain": "OR3 - OR2 [pp]",
                    }[kind], fontsize=7)
                    rax.set_xlabel(var_inst.x_title, fontsize=7)
                    if j == 0:
                        ax.legend(fontsize=5, ncol=2, loc="lower right")
                fig.suptitle(
                    f"{self.config_inst.name} | {variable} | categories {','.join(self.categories)} | "
                    f"in-support events, N >= {self.min_bin_events} per bin",
                    fontsize=8,
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # the per-sample summary: integrated residual for every dataset
            targets = list(TOPO_CLOSURE_TARGETS)
            fig, axes = plt.subplots(
                1, len(targets), figsize=(3.6 * len(targets), 0.22 * len(datasets) + 2), sharey=True,
            )
            axes = np.atleast_1d(axes)
            ypos = np.arange(len(datasets))
            for j, est in enumerate(targets):
                d = np.array([100 * float(integrated[x]["estimators"][est]["delta"]) for x in datasets])
                err = np.array([100 * float(integrated[x]["estimators"][est]["delta_err"]) for x in datasets])
                colour = ["tab:red" if abs(v) > 3 * s else "tab:blue"
                          for v, s in zip(d, np.where(err > 0, err, np.nan))]
                axes[j].errorbar(d, ypos, xerr=err, fmt="none", ecolor="grey", lw=1)
                axes[j].scatter(d, ypos, s=10, c=colour)
                axes[j].axvline(0.0, color="k", lw=0.8)
                axes[j].set_title(est, fontsize=8)
                axes[j].set_xlabel("integrated emu - stored [pp]", fontsize=7)
            axes[0].set_yticks(ypos)
            axes[0].set_yticklabels(datasets, fontsize=5)
            fig.suptitle(
                f"{self.config_inst.name} | integrated closure per sample | red = |pull| > 3",
                fontsize=8,
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        logger.info(f"wrote {pdf_path}")
