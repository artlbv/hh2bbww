# coding: utf-8

"""
Hist producers for the HH -> bbWW(SL) analysis.

The DL hist producers in :py:mod:`hbw.weight.hp_dih` require the ``trigger_weight`` and
``dy_correction_weight`` columns, neither of which is written by any SL producer, so the SL channel
needs its own set. Everything else is shared with DL through ``default_correction_weights``.
"""

from hbw.weight.default import base
from hbw.weight.hp_dih import default_correction_weights


sl_weight_columns = {
    "stitched_normalization_weight": [],
    **default_correction_weights,
}

sl_default = base.derive("sl_default", cls_dict={"weight_columns": sl_weight_columns})

sl_unstitched = base.derive("sl_unstitched", cls_dict={"weight_columns": {
    "dataset_normalization_weight": [],
    **default_correction_weights,
}})
