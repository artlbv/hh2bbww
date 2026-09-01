# CLAUDE.md

## What this repo is

HH → bbWW analysis built on [columnflow](https://github.com/columnflow/columnflow) (task
framework), [law](https://github.com/riga/law) (luigi wrapper) and
[order](https://github.com/riga/order) (analysis/config/dataset model).

- `hbw/` — the analysis package (config, selection, production, ml, inference, tasks, scripts).
- `modules/` — git submodules: `columnflow` (which itself vendors `law` and `order`) and `cmsdb`
  (campaign/dataset definitions). Never edit these to fix an analysis problem; they are pinned.
- Everything runs as a law task. `cf.*` tasks come from columnflow, `hbw.*` tasks are defined in
  `hbw/tasks/` and registered in the `[modules]` section of `law.cfg`.

## Setup

```shell
source setup.sh dev
```

Answers are cached in `.setups/dev.sh` (gitignored) and the script only prompts when that file
does not exist — to change an answer, edit the file rather than deleting it.

`HBW_LAW_CONFIG` in `.setups/dev.sh` picks which `law.*.cfg` at the repo root becomes
`LAW_CONFIG_FILE`, and that file supplies `default_analysis` / `default_config` /
`default_dataset` for every task invocation. **This checkout uses `law.artur.cfg`**
(HH DL analysis `hbw.analysis.hbw_dl.hbw_dl`, config `c24v15`), which is tuned for CERN — see
below. Run `law index` after adding or renaming a task.

## Configs

Configs are registered lazily in `hbw/analysis/create_analysis.py` via `add_lazy_config`. Each
call registers **two** names: the full one and a limited twin (2 files per dataset, for testing)
where the leading `c` becomes `l`. On branch `HH_HHH_incl_2024` the valid names are:

| full | limited | campaign |
| --- | --- | --- |
| `c17v9` | `l17v9` | Run 2, 2017, nano v9 |
| `c22prev14` | `l22prev14` | 2022 preEE, nano v14 |
| `c22postv14` | `l22postv14` | 2022 postEE, nano v14 |
| `c23prev14` | `l23prev14` | 2023 preBPix, nano v14 |
| `c23postv14` | `l23postv14` | 2023 postBPix, nano v14 |
| `c24v15` | `l24v15` | 2024, nano v15 (+ custom HHH samples) |
| `c25v15` | `l25v15` | 2025, nano v15 — needs a different cmsdb pin, see below |
| `c26v15` | `l26v15` | 2026, nano v15 — needs a different cmsdb pin, see below |

On this branch **only `c24v15` / `l24v15` actually build.** `hbw/config/processes.py`
unconditionally does `config.add_process(config.x.procs.n.ttbb_custom)`, and `ttbb_custom` is
only reachable when the campaign contains the `ttbb_{dl,sl,fh}_powheg` datasets — which only the
2024 campaign has. Every other config dies with
`ValueError: object 'ttbb_custom' not known to index 'UniqueObjectIndex(cls=order.process.Process, ...)'`.
Consequently `cf.CreateDatacards --inference-model default` does not work here either, because
that inference model spans the Run 3 v14 configs.

`c25v15` / `c26v15` need the `run3_2025_nano_v15` and `run3_2026_nano_v15` campaigns, which live
on the `MultiHiggs_Run3` branch of `uhh-cms/cmsdb` — a branch that has *diverged* from the
`HHH_HH_2024` commit this repo pins for `modules/cmsdb`. Using them means moving the submodule
pin; `c24v15` still builds with `MultiHiggs_Run3`, but nothing else has been checked.

`c17` and `l17` are dead names (renamed in upstream `fdab666`). If a task dies with
`ValueError: object 'X' not known to index 'UniqueObjectIndex(cls=order.config.Config, ...)'`,
some `default_config` still points at an old name.

**Before any other task, per config:**

```shell
law run hbw.BuildCampaignSummary --config c24v15 --remove-output 0,a,y
```

The lazy config factory instantiates `BuildCampaignSummary` and refuses to build the config until
that output exists. It always passes the *full* name, so building `c24v15` also unlocks `l24v15`.

`hbw/scripts/test_config.py` is the cheapest way to check that the default config resolves at all,
without going through law.

## CERN specifics

Most of the collaboration works at DESY, so several paths in the shared configs are unreachable
here and `law.artur.cfg` exists to work around them:

- `law.nocert.cfg` (and everything inheriting it, e.g. `law.dl.nocert.cfg`) reads LFNs from
  `/pnfs/desy.de/cms/tier2` — not mounted at CERN. Use a grid proxy and the redirectors instead.
- `law.dl.cfg` pins the `c24v15` outputs to `/data/dust/user/letzerba/...` and
  `wlcg_fs_desy_bletzer`. These are keyed `cfg_c24v15__task_*`, and
  `ConfigTask.get_config_lookup_keys` puts the `cfg_` key *before* the `task_` key, so a child
  config's `task_cf.*` entry cannot override them. That is why `law.artur.cfg` inherits
  `law.cfg` directly.
- `local_fs_run3_2024_nano_custom_v15` points at DESY dust in `law.cfg`; `law.artur.cfg`
  repoints it to the CERN mirror `/afs/cern.ch/user/m/mschrode/public`.
- Outputs go to `$CF_STORE_LOCAL` (EOS), plots to a php-plots area on EOS via `wlcg_fs_plots`.
- A valid proxy (`voms-proxy-init -voms cms`) is needed for anything that reads NanoAOD.

## Git

**Hard rule: never commit to, push to, or check out other revisions in any `uhh-cms` repository.**
That covers `hh2bbww` upstream and the `modules/columnflow` and `modules/cmsdb` submodule clones.
Interact with them only through issues and pull requests. To inspect another revision of a
submodule, `git archive` it into a scratch directory rather than checking it out in place.

- `origin` → `git@github.com:artlbv/hh2bbww.git` (fork), `upstream` → `git@github.com:uhh-cms/hh2bbww.git`.
- Work branches off `upstream/HH_HHH_incl_2024`; PRs to upstream go against `master`.
- The submodule URLs in `.gitmodules` are relative (`../../columnflow/columnflow.git`), so they
  resolve correctly under either owner. After switching branches run
  `git submodule update --init --recursive` — branches pin different `columnflow`/`cmsdb` commits.

## Checks

```shell
tests/run_linting   # flake8 over hbw and tests
tests/run_tests     # unit tests inside the columnar sandbox
```

Style: flake8 with `max-line-length = 120`, double quotes enforced, `E128 E306 E402 E722 E731
W504` ignored (see `.flake8`). Modules start with a `# coding: utf-8` line and a docstring.
