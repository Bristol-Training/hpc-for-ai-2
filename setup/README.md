# Instructor setup

Everything the course needs on Isambard-AI comes from this repository. Clone it into `$PROJECTDIR/hpc-for-ai-2`. The scripts here fill its `data/` folder, which git ignores. The instructor notes (`resources/hpc-ai-2-notes.html`, "Instructor setup") have the same steps with more explanation.

Allow a few weeks before the course for access. Allow an afternoon for the setup itself: most of it is waiting for jobs.

## Quick setup

On a login node, in your account in the course project:

```bash
git clone https://github.com/Bristol-Training/hpc-for-ai-2.git $PROJECTDIR/hpc-for-ai-2
cd $PROJECTDIR/hpc-for-ai-2
mkdir -p data/headlines        # then copy uk_headlines.json in here (see Headlines below)
bash setup/setup-all.sh        # one job: the build, both dry runs, then permissions
```

`setup-all.sh` needs `uv` on your `PATH`. If you don't have it, install it once with `curl -LsSf https://astral.sh/uv/install.sh | sh`. It submits one job of about an hour, whose log `setup-all-<job>.out` ends with `SETUP FINISHED` if everything worked. Then do the JupyterHub dry run. Nothing in the repository names a project: every path comes from `$PROJECTDIR` or the clone's own location.

The checklist below does the same steps one at a time.

## What each script makes

| Script | Makes | Time |
|---|---|---|
| `setup-all.sh` | One job that runs the build, then the QLoRA and LoRA dry runs with `publish`, then sets the permissions | About 1 hour |
| `build-environment.sh` | `data/.venv` (PyTorch, Transformers, TRL, PEFT, bitsandbytes, vLLM); Mistral 7B v0.3 and Alpaca in `data/hf-cache`; `data/alpaca/train.parquet`; the **Fine-tuning + Nsight** kernel and `install-kernels.sh` in `data/kernels`. It checks that everything imports, can see the GPU and loads offline | About 30 min |
| `headlines/hear_uk_headlines.py` | `uk_headlines.json` for section 8 (see `headlines/README.md`). It needs internet access, and a rebuild gives slightly different data, because The Hear's archive grows | About 1 hour |
| `dry-run/dry-run.sbatch <mode> publish` | Runs the whole notebook as a participant would; with `publish`, copies the trained adapter, `stats.json` and Nsight reports into `data/fallback/` | About 15 min per mode |
| `profile-summary.sh` | Nothing: participants run it in section 10 to summarise a report | |

## Checklist

Run these over SSH on a login node, from the top of the repository. Each job uses the `interactive` reservation, which allows one job per person at a time, so submit each when the one before has finished.

- [ ] **Access.** The training project exists and has GPU allocation for 2-hour, 1-GH200 sessions for everyone, the instructors and the dry run. Attendees are invited and have set a UNIX username.
- [ ] **Clone the repository:**

    ```bash
    git clone https://github.com/Bristol-Training/hpc-for-ai-2.git $PROJECTDIR/hpc-for-ai-2
    cd $PROJECTDIR/hpc-for-ai-2
    ```

- [ ] **Build the environment, model, dataset and kernel.** You need `uv` on your `PATH` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

    ```bash
    sbatch --reservation=interactive --nodes=1 --gpus=1 --time=01:00:00 \
           --output=setup-%j.out setup/build-environment.sh
    ```

    The end of `setup-<job>.out` should show the package versions, the GPU name, "Offline load of the course copy works: 52002 examples" and the `install-kernels.sh` command.
- [ ] **Headlines:** put `uk_headlines.json` in `data/headlines/`, either from a previous course or by running `headlines/hear_uk_headlines.py` there. Projects can't read each other's folders, so copy it through your own computer, for example:

    ```bash
    scp <old-project-login>:<old-clone>/data/headlines/uk_headlines.json .
    scp uk_headlines.json <new-project-login>:<new-clone>/data/headlines/
    ```

- [ ] **Fallback adapters and reports**, one job after the other:

    ```bash
    sbatch --reservation=interactive setup/dry-run/dry-run.sbatch qlora publish
    sbatch --reservation=interactive setup/dry-run/dry-run.sbatch lora publish
    ```

    Each log (`dry-run-<name>-<job>.out`) should end with `COURSE FLOW PASSED` and list what it published. `data/fallback/` should then hold `lora/` and `qlora/` (each with `final/` and `stats.json`) and six reports in `reports/`.
- [ ] **Permissions,** so the project can read everything and change nothing:

    ```bash
    chmod -R g+rX,g-w $PROJECTDIR/hpc-for-ai-2
    ```

- [ ] **Dry run in JupyterHub,** with 1 GH200, as an attendee would: install the kernel, copy the notebook, run every cell in both modes, and once with `USE_FALLBACK = True`. Fill in the open items of the dry-run list in the instructor notes.

## Keeping it up to date

- Update the course with `git pull`. Don't run `git clean -xdf` or delete the clone: that deletes `data/`.
- After changing the training settings in the notebook, re-run both `dry-run.sbatch ... publish` jobs, so the fallback adapters match.
- After changing the package versions in `build-environment.sh`, re-run it, then the dry runs.
