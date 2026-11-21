"""Run the whole course notebook the way a participant does.

Every code cell runs in order on the course kernel (Fine-tuning + Nsight), and the
notebook's nsys commands run from a separate login shell, like the JupyterHub terminal,
recording the six reports used in section 10. Saves an executed copy of the notebook in
your WORK folder and prints each cell's time.

    python run_notebook.py <qlora|lora> [fallback]

`fallback` skips training and uses the shared fallback adapter, as someone who falls
behind does. Run it through dry-run.sbatch, which installs the kernel first.
"""
import json
import os
import subprocess
import sys
import threading
import time

from jupyter_client.manager import KernelManager

MODE = sys.argv[1]
FALLBACK = sys.argv[2:] == ["fallback"]  # skip training, as someone who falls behind does
USER = os.environ["USER"]
WORK = f"{os.environ['PROJECTDIR']}/{USER}/llm-fine-tuning"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = json.load(open(os.path.join(REPO, "notebooks", "hpc-for-ai-2.ipynb")))
SLICE_AFTER, SLICE_FOR = 90, 20  # seconds into training; length of the slice


def terminal(command):
    """One command in a fresh login shell, set up as the notebook tells people to."""
    setup = "module load cudatoolkit/24.11_12.6; W=$PROJECTDIR/$USER/llm-fine-tuning; "
    r = subprocess.run(["bash", "-lc", setup + command], capture_output=True, text=True)
    print(f"$ {command} -> {r.returncode}\n{r.stdout}{r.stderr}".rstrip(), flush=True)
    assert r.returncode == 0, "terminal command failed"


def start(name):
    terminal(f"nsys start --session=nb-$USER -o $W/{name} --force-overwrite=true")
    time.sleep(3)  # recording is not live the instant `nsys start` returns


def stop():
    terminal("nsys stop --session=nb-$USER")


def record_slice(name):
    time.sleep(SLICE_AFTER)
    start(name)
    time.sleep(SLICE_FOR)
    stop()


def execute(kc, code):
    """Run one cell, returning its outputs in notebook format."""
    outputs, msg_id = [], kc.execute(code)
    while True:
        msg = kc.get_iopub_msg(timeout=3600)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        t, c = msg["msg_type"], msg["content"]
        if t == "stream":
            outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
        elif t in ("execute_result", "display_data"):
            out = {"output_type": t, "data": c["data"], "metadata": c["metadata"]}
            if t == "execute_result":
                out["execution_count"] = c["execution_count"]
            outputs.append(out)
        elif t == "error":
            outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                            "traceback": c["traceback"]})
        elif t == "status" and c["execution_state"] == "idle":
            return outputs


km = KernelManager(kernel_name="finetune-nsys")
km.start_kernel()
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=300)

failed, times = False, []
for i, cell in enumerate(NB["cells"]):
    if cell["cell_type"] != "code":
        continue
    code = "".join(cell["source"]).replace('MODE = "qlora"', f'MODE = "{MODE}"')
    if FALLBACK:
        if "SFTTrainer(model=" in code or 'nvtx.range("train")' in code:
            continue
        code = code.replace("USE_FALLBACK = False", "USE_FALLBACK = True")
    label = next((x for x in ("generate-hf", "generate-vllm", "classify-loop", "classify-batch")
                  if f'nvtx.range("{x}")' in code), None)
    slicer = None
    if label:
        start(label)
    elif 'nvtx.range("train")' in code:
        slicer = threading.Thread(target=record_slice, args=(f"slice-{MODE}",))
        slicer.start()

    began = time.perf_counter()
    cell["outputs"] = execute(kc, code)
    seconds = time.perf_counter() - began
    times.append((i, seconds, code.split("\n")[0][:60]))
    print(f"[cell {i}] {seconds:.1f} s  {times[-1][2]}", flush=True)

    if slicer:
        slicer.join()
    if label:
        stop()
    for out in cell["outputs"]:
        if out["output_type"] == "error":
            print(f"ERROR in cell {i}: {out['ename']}: {out['evalue']}", flush=True)
            failed = True
    if failed:
        break

with open(f"{WORK}/notebook-test-{MODE}{'-fallback' if FALLBACK else ''}.ipynb", "w") as f:
    json.dump(NB, f, indent=1)

print("\n=== cell times", *[f"{i:3d} {s:8.1f} s  {c}" for i, s, c in times], sep="\n", flush=True)
kc.stop_channels()
km.shutdown_kernel(now=False)
if failed:
    sys.exit("COURSE FLOW FAILED")
print("COURSE FLOW PASSED", flush=True)
