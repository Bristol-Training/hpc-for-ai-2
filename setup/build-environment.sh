#!/bin/bash
# Instructor setup for the fine-tuning course on Isambard-AI JupyterHub.
#
# Builds ONE shared Python environment in the course project that holds both the
# training libraries (TRL, PEFT, bitsandbytes) and vLLM, so a single kernel can
# fine-tune and then serve the result. Pre-downloads the model and dataset, and
# creates the Jupyter kernel that participants use:
#
#   "Fine-tuning + Nsight"  started under `nsys launch`, so a
#                           participant can record a profile with `nsys start/stop`
#                           from a JupyterHub terminal.
#
# Everything goes in data/ inside this repository (git ignores it). Run once, as the
# instructor, from the top of the repository, in a GPU job (the import check at the end
# of step 1 needs a GPU):
#   sbatch --reservation=interactive --nodes=1 --gpus=1 --time=01:00:00 \
#          --output=setup-%j.out setup/build-environment.sh
#
# Participants must be members of the same project to read the shared files.

set -euo pipefail

# ---- Edit these -------------------------------------------------------------
# The shared data folder: data/ in the repository this job was submitted from.
DATA=${DATA:-${SLURM_SUBMIT_DIR:-$PWD}/data}
# Participants' own folders go in the course project's folder: $PROJECTDIR/$USER.
# Run this from your account in the course project, so $PROJECTDIR is that project.
USER_ROOT=${USER_ROOT:-${PROJECTDIR:?PROJECTDIR is not set: run this on Isambard-AI}}
MODEL_ID=mistralai/Mistral-7B-v0.3
# -----------------------------------------------------------------------------

mkdir -p "${DATA}"/{kernels,hf-cache,alpaca}

# 1. Python environment: the tutorial's training pins AND its vLLM pin in one place.
#    vllm 0.24.0 requires exactly torch==2.11.0, the same version the training pins
#    use, and trl 1.10.0 supports vllm 0.17-0.26. These resolve together on aarch64.
#    Packages come from the CUDA 12.9 builds the tutorial uses for vLLM on Isambard.
uv venv --clear --seed --python 3.12 "${DATA}/.venv"
source "${DATA}/.venv/bin/activate"
uv pip install \
  torch==2.11.0 \
  transformers==5.15.1 \
  trl==1.10.0 \
  peft==0.20.0 \
  bitsandbytes==0.50.1 \
  accelerate==1.14.0 \
  datasets==5.0.1 \
  sympy==1.14.0 \
  vllm==0.24.0 \
  pandas \
  matplotlib \
  ipykernel \
  --extra-index-url https://wheels.vllm.ai/0.24.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

# Fail now, not on the day, if anything in the combined environment won't import
# or can't see the GPU.
python -c "
import torch, transformers, trl, peft, bitsandbytes, vllm
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0))
print('transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__)
print('bitsandbytes', bitsandbytes.__version__, '| vllm', vllm.__version__)
"

# 2. Pre-download the model and the dataset into the shared, read-only course folder,
#    so nobody downloads anything on the day. Both are ungated: no Hugging Face login.
#    The whole repository, including the original-format copy of the weights: vLLM refuses
#    to load an offline snapshot with any file missing.
export HF_HUB_CACHE="${DATA}/hf-cache"
hf download "${MODEL_ID}"
hf download tatsu-lab/alpaca --repo-type dataset

# Offline, `datasets` cannot find a Hub dataset by name for anyone whose own datasets
# cache is empty, even when its files are in the shared cache. So save one plain
# Parquet file that the notebook loads by path, and check that works the way
# participants will run it: offline, with an empty personal cache.
HF_HUB_OFFLINE=1 python -c "
import glob
from datasets import load_dataset
files = glob.glob('${DATA}/hf-cache/datasets--tatsu-lab--alpaca/snapshots/*/data/train-*.parquet')
load_dataset('parquet', data_files=files, split='train').to_parquet('${DATA}/alpaca/train.parquet')
"
TEST_CACHE=$(mktemp -d)
HF_HUB_OFFLINE=1 HF_DATASETS_CACHE="${TEST_CACHE}" python -c "
from datasets import load_dataset
ds = load_dataset('parquet', data_files='${DATA}/alpaca/train.parquet', split='train')
print('Offline load of the course copy works:', len(ds), 'examples')
"
rm -rf "${TEST_CACHE}"
deactivate

# 3. Record what the CUDA toolkit module provides, so the kernel does not depend on
#    `module load` working inside it: nsys for profiling, and the CUDA toolkit that
#    vLLM may need to compile kernels the first time it runs.
# `module` is not defined in a batch job's non-login shell; load the profile first.
set +u; source /etc/profile >/dev/null 2>&1 || true; module load cudatoolkit/24.11_12.6; set -u
NSYS=$(command -v nsys)
NVCC=$(command -v nvcc || true)
CUDA_TOOLKIT=${CUDA_HOME:-${NVCC:+$(dirname "$(dirname "${NVCC}")")}}
# The headers the module adds (curand.h is in a separate math_libs folder): FlashInfer,
# used by vLLM, compiles its sampling kernels with them the first time vLLM starts.
CUDA_CPATH=${CPATH:-}
echo "Using ${NSYS}: $(${NSYS} --version)"
echo "CUDA toolkit: ${CUDA_TOOLKIT:-not found}"

# 4. Start-up script for the kernel. Jupyter passes the connection file as $1.
cat > "${DATA}/kernels/start-kernel.sh" << EOF
#!/bin/bash
export USER=\${USER:-\$(id -un)}
# Participants' files: readable by the project, changeable only by their owner.
umask 022
# A TMPDIR inherited from the login node may not exist here, and Nsight crashes on it.
[ -d "\${TMPDIR:-/tmp}" ] && [ -w "\${TMPDIR:-/tmp}" ] || export TMPDIR=/tmp
# Shared and read-only: the downloaded model and dataset files.
export HF_HUB_CACHE="${DATA}/hf-cache"
export HF_HUB_OFFLINE=1
# Personal and writable: the prepared dataset and the trainer's tokenised copy.
export HF_DATASETS_CACHE="${USER_ROOT}/\${USER}/.cache/datasets"
export WORK_PATH="${USER_ROOT}/\${USER}/llm-fine-tuning"
mkdir -p "\${WORK_PATH}" "\${HF_DATASETS_CACHE}"

# For vLLM: the CUDA toolkit and compilers the tutorial's serve_vllm.sh sets up.
if [ -n "${CUDA_TOOLKIT}" ]; then
  export CUDA_HOME="${CUDA_TOOLKIT}"
  export PATH="${CUDA_TOOLKIT}/bin:\${PATH}"
  export CPATH="${CUDA_CPATH}\${CPATH:+:\${CPATH}}"
fi
# vLLM compiles kernels on first start with tools from the course environment (ninja).
export PATH="${DATA}/.venv/bin:\${PATH}"
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
# Keep vLLM's compiled-kernel cache personal, since the shared folder is read-only.
export VLLM_CACHE_ROOT="${USER_ROOT}/\${USER}/.cache/vllm"

# Only one kernel per person and node can own the Nsight session "nb-\${USER}". The lock
# and the owner's PID live in this node's /tmp, so a second notebook can tell whether
# the owner is still running.
LOCK="/tmp/nsys-nb-\${USER}.lock"
OWNER="/tmp/nsys-nb-\${USER}.pid"
if [ "\${WITH_NSYS:-0}" = "1" ]; then
  exec 9>"\${LOCK}"
  if flock -n 9; then
    echo \$\$ > "\${OWNER}"
  else
    if kill -0 "\$(cat "\${OWNER}" 2>/dev/null)" 2>/dev/null; then
      # Another notebook has the profiled kernel: run this one without Nsight, rather
      # than end the other one's session.
      echo "start-kernel: another notebook already has the Fine-tuning + Nsight kernel on this node; this one runs without profiling" >&2
      exec 9>&-
      export WITH_NSYS=0
    fi
  fi
fi
if [ "\${WITH_NSYS:-0}" = "1" ]; then
  # The owner was killed (as a restart does), but its Python may still hold the
  # session and the lock: end the session, and wait until both are gone, or the new
  # launch fails.
  "${NSYS}" shutdown --session="nb-\${USER}" >/dev/null 2>&1 || true
  if ! flock -w 30 9; then
    echo "start-kernel: the previous Nsight kernel did not stop; this one runs without profiling" >&2
    exec 9>&-
    exec "${DATA}/.venv/bin/python" -m ipykernel_launcher -f "\$1"
  fi
  for _ in \$(seq 30); do
    "${NSYS}" sessions list 2>/dev/null | grep -qw "nb-\${USER}" || break
    sleep 1
  done
  # A killed nsys can leave its session name registered, and every later launch is
  # then refused that name. With no nsys running on this node, the registry is stale.
  if ! pgrep -x nsys >/dev/null; then
    rm -rf "\${TMPDIR:-/tmp}"/nvidia/nsight_systems/NSys-SessionManager* \
           "\${TMPDIR:-/tmp}"/nvidia/nsight_systems/quadd_session_* /dev/shm/sem.NSys-* 2>/dev/null
  fi
  echo \$\$ > "\${OWNER}"
  # Attach Nsight but record nothing until "nsys start --session=nb-\${USER}".
  # vLLM runs the model in a separate engine process and replays work as CUDA
  # graphs: follow child processes, and show each kernel inside the graphs.
  exec "${NSYS}" launch --session="nb-\${USER}" --trace=cuda,nvtx --cuda-memory-usage=true \\
       --trace-fork-before-exec=true --cuda-graph-trace=node \\
       "${DATA}/.venv/bin/python" -m ipykernel_launcher -f "\$1"
else
  exec "${DATA}/.venv/bin/python" -m ipykernel_launcher -f "\$1"
fi
EOF
chmod 755 "${DATA}/kernels/start-kernel.sh"

# 5. The kernel spec that participants link into ~/.local/share/jupyter/kernels/.
mkdir -p "${DATA}/kernels/finetune-nsys"

cat > "${DATA}/kernels/finetune-nsys/kernel.json" << EOF
{
  "argv": ["${DATA}/kernels/start-kernel.sh", "{connection_file}"],
  "display_name": "Fine-tuning + Nsight",
  "language": "python",
  "env": {"WITH_NSYS": "1"}
}
EOF

# 6. What participants run once to see the kernel in JupyterLab. A link, not a copy,
#    so any later change to the shared kernel reaches everyone.
cat > "${DATA}/kernels/install-kernels.sh" << EOF
#!/bin/bash
# Make the course kernel appear in your JupyterLab kernel menu. Safe to run again.
set -e
mkdir -p ~/.local/share/jupyter/kernels
rm -rf ~/.local/share/jupyter/kernels/finetune ~/.local/share/jupyter/kernels/finetune-nsys
ln -s "${DATA}/kernels/finetune-nsys" ~/.local/share/jupyter/kernels/finetune-nsys
echo "Installed the Fine-tuning + Nsight kernel. Reload JupyterLab to see it."
EOF
chmod 755 "${DATA}/kernels/install-kernels.sh"

# 7. Let everyone in the project read and run it. Nothing shared is writable.
chmod -R g+rX,g-w "${DATA}"

echo
echo "Done. Participants run once, in a JupyterHub terminal:"
echo "  bash ${DATA}/kernels/install-kernels.sh"
