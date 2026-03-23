#!/usr/bin/env bash
# =============================================================================
# Enumerate (ISL, OSL) and CONC combinations, then dispatch each to
# run_local_kimi_mi355x.sh.
# Usage:
#   bash launch_task.sh
# =============================================================================

set -x


SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

EP_LIST=(1)
CONC_LIST=(4 8 16 32 64 128)
ISL_OSL_LIST=(
    "1024 1024"
    "8192 1024"
)

TOTAL=${#CONC_LIST[@]}
TOTAL=$(( TOTAL * ${#ISL_OSL_LIST[@]} ))
TOTAL=$(( TOTAL * ${#EP_LIST[@]} ))
RUN_IDX=0
START_FROM=${START_FROM:-1}

for ep in "${EP_LIST[@]}"; do
    for isl_osl in "${ISL_OSL_LIST[@]}"; do
        read -r isl osl <<< "$isl_osl"
        for conc in "${CONC_LIST[@]}"; do
            RUN_IDX=$(( RUN_IDX + 1 ))
            echo ""
            echo "###############################################################"
            echo "# Run ${RUN_IDX}/${TOTAL}: EP=${ep} ISL=${isl} OSL=${osl} CONC=${conc}"
            echo "###############################################################"
            echo ""

            EP_SIZE="$ep" ISL="$isl" OSL="$osl" CONC="$conc" \
                bash "${SCRIPT_DIR}/run_local_kimi_mi355x.sh" 1>0.log 2>&1
                
            sleep 3s
            bash kill_vllm_server.sh
            sleep 3s
            bash kill_vllm_server.sh
            sleep 3s
        done
    done
done

echo ""
echo "All ${TOTAL} benchmark runs completed."
