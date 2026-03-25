#!/bin/bash
# Use existing CARLA_ROOT if set; otherwise fallback to container default
export CARLA_ROOT=${CARLA_ROOT:-/workspace/carla}

if [ ! -d "${CARLA_ROOT}/PythonAPI" ]; then
	echo "[ERROR] Invalid CARLA_ROOT: ${CARLA_ROOT}" >&2
	echo "        Set CARLA_ROOT to your CARLA directory (must contain PythonAPI)." >&2
	exit 1
fi

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10:-0}
export EXTERNAL_CARLA=${EXTERNAL_CARLA:-False}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

EXTERNAL_CARLA_FLAG=""
if [ "${EXTERNAL_CARLA}" = "True" ] || [ "${EXTERNAL_CARLA}" = "true" ] || [ "${EXTERNAL_CARLA}" = "1" ]; then
    EXTERNAL_CARLA_FLAG="--external-carla"
fi

# ── Auto-retry loop ──
# CARLA (UE4) occasionally crashes with Signal 11 (SIGSEGV).
# When that happens, leaderboard_evaluator.py exits with code != 0.
# We restart CARLA and resume from the last checkpoint automatically.
MAX_RETRIES=${MAX_RETRIES:-10}
RETRY_WAIT=${RETRY_WAIT:-30}

# Disable errexit so we can capture the exit code of the python process
set +e

for (( attempt=1; attempt<=MAX_RETRIES; attempt++ )); do
    echo "[run_evaluation.sh] Attempt ${attempt}/${MAX_RETRIES} on GPU ${GPU_RANK}, port ${PORT}"

    CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
        --routes=${ROUTES} \
        --repetitions=${REPETITIONS} \
        --track=${CHALLENGE_TRACK_CODENAME} \
        --checkpoint=${CHECKPOINT_ENDPOINT} \
        --agent=${TEAM_AGENT} \
        --agent-config=${TEAM_CONFIG} \
        --debug=${DEBUG_CHALLENGE} \
        --record=${RECORD_PATH} \
        --resume=${RESUME} \
        --port=${PORT} \
        --traffic-manager-port=${TM_PORT} \
        --gpu-rank=${GPU_RANK} \
        ${EXTERNAL_CARLA_FLAG}

    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[run_evaluation.sh] GPU ${GPU_RANK}: All routes completed successfully."
        break
    fi

    echo "[run_evaluation.sh] GPU ${GPU_RANK}: Evaluator exited with code ${EXIT_CODE} (attempt ${attempt}/${MAX_RETRIES})."

    if [ ${attempt} -eq ${MAX_RETRIES} ]; then
        echo "[run_evaluation.sh] GPU ${GPU_RANK}: Max retries reached. Giving up."
        exit 1
    fi

    # If using external CARLA, wait for the server to be restarted manually or by a watchdog.
    # If CARLA is managed internally, it will be relaunched by leaderboard_evaluator.py itself.
    if [ -n "${EXTERNAL_CARLA_FLAG}" ]; then
        echo "[run_evaluation.sh] GPU ${GPU_RANK}: Waiting for external CARLA on port ${PORT} to come back..."
        for (( w=0; w<120; w++ )); do
            if timeout 2 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
                echo "[run_evaluation.sh] GPU ${GPU_RANK}: CARLA on port ${PORT} is back."
                break
            fi
            sleep 5
        done
        if ! timeout 2 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
            echo "[run_evaluation.sh] GPU ${GPU_RANK}: CARLA on port ${PORT} not reachable after 600s. Giving up."
            exit 1
        fi
    fi

    echo "[run_evaluation.sh] GPU ${GPU_RANK}: Resuming in ${RETRY_WAIT}s..."
    sleep ${RETRY_WAIT}
done
