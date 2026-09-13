# `hbw/config/jsons/`

Small configuration files that are **tracked in this repository**, as opposed to
the correction files under `hbw_external_base`.

The distinction is not about file format, it is about what the file does:

- **Under `hbw_external_base`** belong *corrections* — measurements of how the
  detector or the reconstruction behaves. They are often large (the JERC files
  are ~3 MB each), they vary by campaign and by point in time, and they are
  produced and validated elsewhere.
- **Here** belong files that are part of the *definition of the analysis*: small,
  not campaign-specific, and changing one changes what the analysis means rather
  than how accurately it is corrected.

A file in this directory should be reviewable in a pull request, should appear in
`git log`, and should be citable by commit. If that is not true of it, it probably
belongs under `hbw_external_base` instead.

## Contents

### `mc_event_splitter.json.gz`

Assigns every simulated 2024 event to exactly one of 2024, 2025 or 2026, by
hashing its event number to a variate uniform on the unit interval and comparing
against fixed thresholds:

```
2024 * (x < 0.43697) + 2025 * (x >= 0.43697) * (x < 0.88228) + 2026 * (x >= 0.88228)
```

giving shares of 43.697 %, 44.531 % and 11.772 %, chosen to match the recorded
integrated luminosity of each year.

The Summer24 samples serve all three periods. Without the partition, an analysis
of each period would use the same simulated events and the three results would
carry correlated statistical uncertainties that no later combination could undo.
The rule depends on the event number alone and on no physics quantity, so the
three subsets are statistically equivalent rather than merely disjoint.

It is applied in `hbw/config/config_run2.py` when NanoAOD is read, so events of
other years never enter the processing chain.

**This file decides which events an analysis is allowed to use.** Editing it
changes every downstream number, in every campaign from 2024 on, silently. That
is why it is here and not in a directory where a file can be replaced in place.

Provenance: <https://gist.github.com/riga/f9476f3b1477f1609683bea68ae64897>, with
frequencies from the CMS public luminosity results.

## A note on `.gitignore`

The repository ignores `*.json` and `*.yaml`, and a user-level global ignore of
`*.gz` is common. Both can silently drop a file you add here — `git add` will
simply do nothing and say nothing. The `!hbw/config/jsons/` negation in
`.gitignore` guards the first case; if a file still refuses to stage, check
`git check-ignore -v <path>` against your global config before assuming it worked.
