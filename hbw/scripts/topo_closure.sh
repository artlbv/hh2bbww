#!/usr/bin/env bash

# Driver for the TOPO emulation closure test (hbw.TopoEmulationClosure).
#
#   source setup.sh dev
#   bash hbw/scripts/topo_closure.sh limited     # l24v15, 2 files per dataset, local workers
#   bash hbw/scripts/topo_closure.sh full        # c24v15, every file, HTCondor
#   bash hbw/scripts/topo_closure.sh limited hh_ggf_hbb_hvvqqlnu_kl1_kt1_powheg   # one dataset
#
# The dataset list is the same 50 the sensitivity study runs on: signal + tt + st + w_lnu + dy +
# qcd_mu + VV, with the electron-enriched QCD deliberately absent. Keep it in sync with the study,
# otherwise the closure is measured on a different population than the one it is used to correct.
#
# The default category is "incl". Note what the category axis actually carries for sl1_topo: the
# leaves are the individual blocks (incl, sr, 1mu, 1e, highmet, lowmet, fake), NOT composites like
# sr__1mu, because the composite categories come from the pre_ml_cats producer which this chain does
# not run. Ask for "1mu" to get the muon channel.

set -euo pipefail

MODE="${1:-limited}"
shift || true

ANALYSIS="hbw.analysis.hbw_sl.hbw_sl"
SELECTOR="sl1_topo"
PRODUCERS="topo_or3_weights"
HIST_PRODUCER="topo_closure"
CATEGORIES="${TOPO_CLOSURE_CATEGORIES:-incl}"

DATASETS="dy_ee_m10to50_amcatnlo,dy_ee_m50toinf_0j_amcatnlo,dy_ee_m50toinf_1j_amcatnlo,\
dy_ee_m50toinf_2j_amcatnlo,dy_ee_m50toinf_amcatnlo,dy_mumu_m10to50_amcatnlo,\
dy_mumu_m50toinf_0j_amcatnlo,dy_mumu_m50toinf_1j_amcatnlo,dy_mumu_m50toinf_2j_amcatnlo,\
dy_mumu_m50toinf_amcatnlo,dy_tautau_m10to50_amcatnlo,dy_tautau_m50toinf_0j_amcatnlo,\
dy_tautau_m50toinf_1j_amcatnlo,dy_tautau_m50toinf_2j_amcatnlo,dy_tautau_m50toinf_amcatnlo,\
hh_ggf_hbb_hvvqqlnu_kl1_kt1_powheg,qcd_mu_pt20to30_pythia,qcd_mu_pt30to50_pythia,\
qcd_mu_pt50to80_pythia,qcd_mu_pt80to120_pythia,qcd_mu_pt120to170_pythia,qcd_mu_pt170to300_pythia,\
qcd_mu_pt300to470_pythia,qcd_mu_pt470to600_pythia,qcd_mu_pt600to800_pythia,\
qcd_mu_pt800to1000_pythia,qcd_mu_pt1000toinf_pythia,st_schannel_t_lep_4f_amcatnlo,\
st_schannel_tbar_lep_4f_amcatnlo,st_twchannel_t_dl_powheg,st_twchannel_t_fh_powheg,\
st_twchannel_t_sl_powheg,st_twchannel_tbar_dl_powheg,st_twchannel_tbar_fh_powheg,\
st_twchannel_tbar_sl_powheg,tt_dl_powheg,tt_fh_powheg,tt_sl_powheg,w_lnu_1j_madgraph,\
w_lnu_2j_madgraph,w_lnu_3j_madgraph,w_lnu_4j_madgraph,ww_dl_powheg,ww_sl_powheg,\
wz_wlnu_zll_powheg,wz_wlnu_zqq_powheg,wz_wqq_zll_powheg,zz_zll_zll_powheg,zz_zll_znunu_powheg,\
zz_zqq_zll_powheg"

# every process the 50 span, so the task's processes parameter resolves without a config group
PROCESSES="hh_ggf_hbb_hvvqqlnu_kl1_kt1,tt,st,w_lnu,dy,qcd_mu,vv"

if [ $# -gt 0 ]; then
    DATASETS="$1"
    shift
fi

case "$MODE" in
    limited)
        CONFIG="l24v15"
        # The WLCG read cache is off by default for local multi-worker runs. law's cache
        # allocate() lists the cache directory and then stats each entry, so once the cache is at
        # its max size and eviction runs on every allocation, two workers race and the loser dies
        # with FileNotFoundError on a path that existed a moment earlier. Streaming from the
        # redirector removes the race; set TOPO_CLOSURE_CACHE=true to put it back.
        export CF_WLCG_USE_CACHE="${TOPO_CLOSURE_CACHE:-false}"
        EXTRA=(--workers "${TOPO_CLOSURE_WORKERS:-4}")
        ;;
    full)
        CONFIG="c24v15"
        # every upstream workflow goes to the batch; the closure task itself is a single cheap
        # reduce over merged histograms and stays local.
        EXTRA=(
            --workers "${TOPO_CLOSURE_WORKERS:-12}"
            --cf.CalibrateEvents-workflow htcondor
            --cf.SelectEvents-workflow htcondor
            --cf.ReduceEvents-workflow htcondor
            --cf.ProduceColumns-workflow htcondor
            --cf.CreateHistograms-workflow htcondor
        )
        ;;
    *)
        echo "usage: $0 {limited|full} [dataset[,dataset...]]" >&2
        exit 1
        ;;
esac

set -x
law run hbw.TopoEmulationClosure \
    --analysis "$ANALYSIS" \
    --configs "$CONFIG" \
    --selector "$SELECTOR" \
    --producers "$PRODUCERS" \
    --hist-producer "$HIST_PRODUCER" \
    --categories "$CATEGORIES" \
    --datasets "$DATASETS" \
    --processes "$PROCESSES" \
    "${EXTRA[@]}" \
    "$@"
