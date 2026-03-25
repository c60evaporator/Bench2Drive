#!/bin/bash
# =============================================================================
# Multi-GPU parallel evaluation for UniAD on Bench2Drive.
#
# Reads EVAL_GPUS and port settings from the .env file (via docker-compose
# environment) to determine which GPUs and ports to use. The port rule is
# shared with carla/launch_carla_servers_host.sh:
#   CARLA RPC port = CARLA_BASE_PORT + index * CARLA_PORT_STEP
#
# Usage (inside bench2drive container):
#   cd /workspace/Bench2Drive
#   bash leaderboard/scripts/run_evaluation_multi_uniad.sh
#
# Prerequisites:
#   - CARLA servers are running (via carla/launch_carla_servers.sh)
#   - EXTERNAL_CARLA env var is set to True
# =============================================================================
export EXTERNAL_CARLA=${EXTERNAL_CARLA:-True}

# ── Port settings (match carla/launch_carla_servers.sh) ──
BASE_PORT=${CARLA_BASE_PORT:-30000}
BASE_TM_PORT=${CARLA_BASE_TM_PORT:-50000}
PORT_STEP=${CARLA_PORT_STEP:-150}

# ── GPU list from EVAL_GPUS ──
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${EVAL_GPUS}"
NUM_GPUS=${#GPU_ARRAY[@]}

# ── Evaluation settings ──
IS_BENCH2DRIVE=True
PLANNER_TYPE=traj
ALGO=uniad
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=leaderboard/team_code/uniad_b2d_agent.py
# Must set YOUR_CKPT_PATH
TEAM_CONFIG=Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+Bench2DriveZoo/ckpts/uniad_base_b2d.pth
BASE_CHECKPOINT_ENDPOINT=Bench2DriveZoo/work_dirs/closedloop/${ALGO}_${PLANNER_TYPE}/tmp/eval
SAVE_PATH=Bench2DriveZoo/work_dirs/closedloop/${ALGO}_${PLANNER_TYPE}/eval_multi_${ALGO}_${PLANNER_TYPE}

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    TASK_NUM=${NUM_GPUS}  # 1 task per GPU
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

echo "============================================================"
echo " Multi-GPU Evaluation"
echo " EVAL_GPUS            : ${EVAL_GPUS}"
echo " NUM_GPUS             : ${NUM_GPUS}"
echo " BASE_PORT            : ${BASE_PORT}"
echo " PORT_STEP            : ${PORT_STEP}"
echo " EXTERNAL_CARLA       : ${EXTERNAL_CARLA}"
echo "============================================================"

for (( i=0; i<NUM_GPUS; i++ )); do
    PORT=$((BASE_PORT + i * PORT_STEP))
    TM_PORT=$((BASE_TM_PORT + i * PORT_STEP))
    GPU_RANK=${GPU_ARRAY[$i]}
    ROUTES="${BASE_ROUTES}_${i}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}_${i}.json"

    # Ensure the checkpoint output directory exists
    mkdir -p "$(dirname "${CHECKPOINT_ENDPOINT}")"

    echo -e "\033[32m [${i}/${NUM_GPUS}] GPU=${GPU_RANK}  PORT=${PORT}  TM_PORT=${TM_PORT} \033[0m"
    echo -e "\033[32m   ROUTES: ${ROUTES} \033[0m"
    echo -e "\033[32m   CHECKPOINT: ${CHECKPOINT_ENDPOINT} \033[0m"
    echo -e "-----------------------------------------------------------"

    bash leaderboard/scripts/run_evaluation.sh \
        $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG \
        $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \
        2>&1 > ${BASE_ROUTES}_${i}_${ALGO}_${PLANNER_TYPE}.log &
    sleep 5
done
wait
