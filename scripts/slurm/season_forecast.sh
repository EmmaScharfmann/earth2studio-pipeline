#!/bin/bash
# Run a season's forecasts on one node, one process per GPU.
#
#   mkdir -p logs        # Slurm writes the job log there and won't create it
#   sbatch scripts/slurm/season_forecast.sh                                # summer 2026
#   sbatch scripts/slurm/season_forecast.sh configs/summer2026_france.yaml --models aifs-ens1.0
#   sbatch scripts/slurm/season_forecast.sh configs/summer2026_france.yaml --start 2026-07-01 --end 2026-07-31
#
# Arguments after the config are passed to every run_season.py process
# (--models, --start/--end, --season/--year, --limit, ...).
#
# Each GPU runs its share of the jobs (run_season.py --shard i), balanced by
# expected cost so all GPUs finish at about the same time. Resubmitting the
# same command resumes: finished forecasts are skipped, failed ones retried.
# Per-GPU logs: logs/season_forecast_<jobid>_gpu<i>.log
#
#SBATCH --job-name=season-forecast
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/season_forecast_%j.out

set -uo pipefail

CONFIG=${1:-configs/summer2026_france.yaml}
[ $# -gt 0 ] && shift
EXTRA=("$@")

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}" || exit 1
mkdir -p logs

NUM=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "$(date) | job ${SLURM_JOB_ID:-local} on $(hostname), ${NUM} GPU(s), config ${CONFIG} ${EXTRA[*]}"

# Show the plan (and resolve the base environment once, before 8 processes
# would all try to at the same time).
uv run scripts/run_season.py "$CONFIG" --stage status --num-shards "$NUM" "${EXTRA[@]}" || exit 1

pids=()
for ((i = 0; i < NUM; i++)); do
    CUDA_VISIBLE_DEVICES=$i uv run scripts/run_season.py "$CONFIG" --stage forecast \
        --shard "$i" --num-shards "$NUM" "${EXTRA[@]}" \
        > "logs/season_forecast_${SLURM_JOB_ID:-local}_gpu${i}.log" 2>&1 &
    pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "$(date) | GPU ${i}: some forecasts failed (see logs/season_forecast_${SLURM_JOB_ID:-local}_gpu${i}.log)"
        failed=1
    fi
done

echo "$(date) | done"
uv run scripts/run_season.py "$CONFIG" --stage status "${EXTRA[@]}"
exit $failed
