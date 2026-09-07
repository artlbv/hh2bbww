# coding: utf-8

"""
Hist producers for the TOPO trigger sensitivity study.

The three arms of the comparison differ only in which trigger weight multiplies the event, never in
the selection -- the selection is always ``sl1_topo``, which applies no HLT cut at all. That is what
makes the three directly comparable.

* ``sl_topo_none`` -- no trigger weight: the acceptance *ceiling*, i.e. what the analysis would
  select if every event were recorded.
* ``sl_topo_or2``  -- the stored decision of ``HLT_IsoMu24 | HLT_Mu12_IsoVVL_PFHT150_PNetBTag0p53``.
  This is the **denominator of the quoted result** and needs no emulation, because both paths are
  stored bits in Summer24 NanoAODv15.
* ``sl_topo_or3``  -- OR2 plus the emulated TOPO residual. Requires the exported efficiency
  ensemble, so it does not run until that artefact exists.
"""

from hbw.weight.default import base
from hbw.weight.hp_sl import sl_weight_columns


sl_topo_none = base.derive("sl_topo_none", cls_dict={"weight_columns": sl_weight_columns})

sl_topo_or2 = base.derive("sl_topo_or2", cls_dict={"weight_columns": {
    **sl_weight_columns,
    # no shift sources yet: SF_OR2 is 1 until hbw.ComputeTriggerSF measures it
    "topo_trigger_weight_or2": [],
}})

# NOTE: not runnable yet -- topo_trigger_weight comes from the exported efficiency ensemble, which
#       does not exist. Kept here so the comparison structure is complete and reviewable.
sl_topo_or3 = base.derive("sl_topo_or3", cls_dict={"weight_columns": {
    **sl_weight_columns,
    "topo_trigger_weight": [],
}})
