#!/bin/bash
# One-screen summary of an Nsight report: labelled ranges, kernel count, total kernel time,
# and the longest kernels. Give the report's name; it looks in this folder, then in yours:
#   bash $PROJECTDIR/hpc-for-ai-2/setup/profile-summary.sh generate-hf
set -euo pipefail
NSYS=$(command -v nsys) || { echo "Run first: module load cudatoolkit/24.11_12.6"; exit 1; }
R=${1:?Give a report name, for example: bash $0 generate-hf}
R=${R%.nsys-rep}
W=/projects/b6bf/${USER}/llm-fine-tuning
if [ ! -f "${R}.nsys-rep" ]; then
  if [ -f "${W}/${R}.nsys-rep" ]; then
    R="${W}/${R}"
  else
    echo "No report ${R}.nsys-rep in this folder or in ${W}."
    echo "Reports in ${W}: $(cd "${W}" 2>/dev/null && ls -- *.nsys-rep 2>/dev/null | sed 's/\.nsys-rep$//' | tr '\n' ' ')"
    exit 1
  fi
fi
"${NSYS}" stats --force-export=true --report nvtx_sum --format csv "${R}.nsys-rep" 2>/dev/null > /tmp/nvtx.$$ || true
if ! "${NSYS}" stats --report cuda_gpu_kern_sum --format csv "${R}.nsys-rep" > /tmp/kern.$$ 2> /tmp/kern-err.$$; then
  echo "nsys stats could not read ${R}.nsys-rep:"
  cat /tmp/kern-err.$$
  rm -f /tmp/nvtx.$$ /tmp/kern.$$ /tmp/kern-err.$$
  exit 1
fi
python3 - "$(basename "${R}")" /tmp/nvtx.$$ /tmp/kern.$$ <<'PY'
import csv
import sys

def num(s):
    return float(s.replace(",", ""))

def table(path):
    return [r for r in csv.reader(open(path)) if len(r) > 3 and r[0][:1].isdigit()]

name, nvtx, kern = sys.argv[1], table(sys.argv[2]), table(sys.argv[3])
total = sum(num(r[1]) for r in kern) / 1e9
print(f"Report: {name}")
if not kern:
    print("  No GPU kernels: the GPU was idle while this was recording. For a training")
    print("  slice, record while the loss is printing, not after training has finished.")
    sys.exit()
ranges = [r for r in nvtx if "CCCL" not in r[-1]]
for r in ranges:
    print(f"  Range {r[-1].lstrip(':')}: {num(r[1]) / 1e9:.1f} s")
print(f"  Kernels: {sum(int(num(r[2])) for r in kern):,}")
print(f"  Total kernel time: {total:.2f} s")
if ranges:
    print(f"  GPU busy: {100 * total / (num(ranges[0][1]) / 1e9):.0f}% of the range")
print("  Longest kernels:")
for r in kern[:6]:
    print(f"    {num(r[1]) / 1e9:7.3f} s  {int(num(r[2])):>9,} calls  {r[-1][:58]}")
PY
rm -f /tmp/nvtx.$$ /tmp/kern.$$ /tmp/kern-err.$$
