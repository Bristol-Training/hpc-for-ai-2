#!/bin/bash
# Set up the course in a new project with one command. Run it on a login node, from
# your clone of this repository in the course project:
#
#   bash setup/setup-all.sh
#
# It submits one GPU job (the `interactive` reservation accepts only one job per person
# at a time) that runs, in order, stopping at the first failure:
#   1. setup/build-environment.sh    Python environment, model, dataset and kernel (~30 min)
#   2. dry-run.sbatch qlora publish  QLoRA fallback adapter and the supplied reports (~15 min)
#   3. dry-run.sbatch lora publish   LoRA fallback adapter and its report (~15 min)
#   4. chmod, so the project can read everything and change nothing
# Its log is setup-all-<job>.out at the top of the repository.
#
# Set RESERVATION= to submit without a reservation, or to another reservation's name.
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=setup-all-%j.out
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  # On the login node: check, then submit this same script as the job.
  cd "$(dirname "$0")/.."
  [ -n "${PROJECTDIR:-}" ] || { echo "PROJECTDIR is not set: run this on Isambard-AI"; exit 1; }
  command -v uv >/dev/null || { echo "uv is not on your PATH: see setup/README.md"; exit 1; }
  if [ ! -s data/headlines/uk_headlines.json ]; then
    echo "data/headlines/uk_headlines.json is missing: the dry runs need it."
    echo "Copy it from a previous course, or build it (see setup/headlines/README.md), then run this again."
    exit 1
  fi
  RES=${RESERVATION-interactive}
  JOB=$(sbatch --parsable ${RES:+--reservation=$RES} setup/setup-all.sh)
  echo "Submitted job $JOB. Its log is $(pwd)/setup-all-$JOB.out (watch with: squeue --me)."
  echo "It takes about an hour, and ends with 'SETUP FINISHED' if everything worked."
  exit 0
fi

# In the job, from the top of the repository (where it was submitted).
cd "$SLURM_SUBMIT_DIR"
echo "########## 1. build-environment.sh"
bash setup/build-environment.sh
for MODE in qlora lora; do
  echo "########## dry run: $MODE publish"
  bash setup/dry-run/dry-run.sbatch "$MODE" publish
done
echo "########## permissions"
chmod -R g+rX,g-w "$SLURM_SUBMIT_DIR"
echo "SETUP FINISHED. Next: the JupyterHub dry run in setup/README.md."
