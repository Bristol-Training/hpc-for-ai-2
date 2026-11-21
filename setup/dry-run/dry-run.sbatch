#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=dry-run-%x-%j.out
# Dry run of the course, and how the shared fallback/ folder is made.
#
# Runs the whole notebook in one mode on the course kernel, as a participant would,
# recording the six Nsight reports from a terminal. Run it from the top of the repo:
#
#   sbatch --reservation=interactive setup/dry-run/dry-run.sbatch qlora
#   sbatch --reservation=interactive setup/dry-run/dry-run.sbatch lora
#   sbatch --reservation=interactive setup/dry-run/dry-run.sbatch qlora fallback   # skip training
#
# Add `publish` after the mode to copy this run's adapter, stats.json and reports into
# data/fallback/ in this repository, replacing what is there:
#
#   sbatch --reservation=interactive setup/dry-run/dry-run.sbatch qlora publish
#   sbatch --reservation=interactive setup/dry-run/dry-run.sbatch lora publish
#
# The QLoRA run publishes all six reports; the LoRA run publishes only slice-lora.
set -uo pipefail
unset JUPYTER_PATH VIRTUAL_ENV
export TMPDIR=/tmp  # the login node's TMPDIR doesn't exist on compute nodes
MODE=${1:?Give a mode: qlora or lora}
shift
PUBLISH=0; ARGS=()
for a in "$@"; do [ "$a" = publish ] && PUBLISH=1 || ARGS+=("$a"); done
REPO=${SLURM_SUBMIT_DIR:-$PWD}
DATA=$REPO/data
W=$PROJECTDIR/$USER/llm-fine-tuning
[ -f "$REPO/notebooks/hpc-for-ai-2.ipynb" ] || { echo "Submit this from the top of the hpc-for-ai-2 repo"; exit 1; }
if [ $PUBLISH = 1 ] && [ ${#ARGS[@]} -gt 0 ]; then echo "publish needs a full run, not fallback"; exit 1; fi

bash "$DATA/kernels/install-kernels.sh"
# Empty personal caches, as for a participant. Reports go to fresh files.
rm -rf "$PROJECTDIR/$USER/.cache/vllm" "$PROJECTDIR/$USER/.cache/datasets"
rm -f "$W"/*.nsys-rep "$W"/*.sqlite
"$DATA/.venv/bin/python" "$REPO/setup/dry-run/run_notebook.py" "$MODE" "${ARGS[@]}" 2>&1 \
  | grep -v -E '^\[1/1\]|Capturing CUDA|Collecting data'
STATUS=${PIPESTATUS[0]}

echo "########## the section 9 reading commands, in a login shell"
REPO="$REPO" bash -l <<'TERM'
module load cudatoolkit/24.11_12.6
W=$PROJECTDIR/$USER/llm-fine-tuning
cp -n "$REPO/data/fallback/reports/"*.nsys-rep $W/
for r in slice-qlora slice-lora generate-hf generate-vllm classify-loop classify-batch; do
  bash "$REPO/setup/profile-summary.sh" $r
done
TERM

if [ $PUBLISH = 1 ]; then
  if [ "$STATUS" != 0 ]; then echo "Not publishing: the notebook run failed"; exit 1; fi
  F=$DATA/fallback
  mkdir -p "$F/$MODE" "$F/reports"
  rm -rf "$F/$MODE/final"
  cp -r "$W/sft-output/$MODE/final" "$F/$MODE/final"
  cp "$W/sft-output/$MODE/stats.json" "$F/$MODE/stats.json"
  if [ "$MODE" = qlora ]; then
    REPORTS="slice-qlora generate-hf generate-vllm classify-loop classify-batch"
  else
    REPORTS="slice-lora"
  fi
  for r in $REPORTS; do cp "$W/$r.nsys-rep" "$F/reports/"; done
  chmod -R g+rX,g-w "$F"
  echo "Published to $F:"; ls -l "$F/$MODE" "$F/reports"
fi
exit "$STATUS"
