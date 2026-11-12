# Running GPU Jobs on a University HPC Cluster

A practical guide to running any GPU workload (model inference, training,
evaluation) on a Slurm-managed HPC cluster, including how to work with an AI
coding assistant while following university rules on AI use.

Everything cluster-specific is written as a `<placeholder>`. Replace it with
your own value, without the angle brackets.

---

## 1. How an HPC cluster works

An HPC cluster is a group of computers you reach over the network and control
by typing commands. It has three parts:

| Part | What it is | What you do there |
|---|---|---|
| **Login node** | The machine you land on when you connect | Edit files, install software, download data, submit jobs. **No heavy computing.** |
| **Compute nodes** | The machines with the GPUs | You don't log in. Your jobs run here. |
| **Scheduler (Slurm)** | The queue manager | Takes your job requests and runs them when the hardware you asked for is free |

You never run GPU work directly. You write a **job script**, a small text file
that says "I need 1 GPU for 2 hours; run this command". Then you **submit** it.
Slurm queues it, runs it on a compute node when a GPU frees up, and saves
everything it prints to a **log file**. The loop is always:

```
write job script  ->  sbatch (submit)  ->  wait in queue  ->  runs  ->  read log + results
```

The job runs on the cluster, not in your terminal. You can disconnect or close
your laptop, and it carries on.

### Storage

Clusters usually offer at least two storage areas. Check your cluster's
documentation for the exact paths and quotas.

| Area | Typical path | Backed up? | Use it for |
|---|---|---|---|
| **Home** | `/home/$USER` | Yes | Code, job scripts, small results, software environments |
| **Scratch** | `/scratch/$USER` | **No** | Large downloads (model weights, datasets), big intermediate outputs |

Rule of thumb: anything you can re-download goes on scratch. Anything you
couldn't recreate goes in home, or in git.

---

## 2. The working method: you run the cluster, the AI edits code locally

Many universities restrict how AI tools may be used with shared HPC systems.
Typical rules forbid AI tools from:

- receiving passwords, SSH private keys, API tokens or MFA codes
- being given infrastructure details (hostnames, IP addresses, node names,
  partition layouts)
- running autonomously on the cluster
- running commands that no human has reviewed

The workflow below keeps the AI assistant completely off the cluster, with
git as the only link between the two.

```
        YOUR LAPTOP                         GITHUB                      HPC CLUSTER
 ┌──────────────────────────┐                                  ┌──────────────────────────┐
 │ AI assistant edits code, │                                  │ YOU type every command:  │
 │ tests it locally,        │   git push     ┌──────────┐      │  git pull                │
 │ commits it               │ ─────────────► │   repo   │ ───► │  sbatch / squeue / ...   │
 │                          │  (you run it)  └──────────┘      │  read logs               │
 │ You review and push      │                                  │                          │
 └──────────────────────────┘                                  └──────────────────────────┘
             ▲                                                              │
             └──────── results: VS Code "Download" / you copy them ◄────────┘
             └──────── errors: you paste the relevant lines, hostnames removed
```

### Who does what

| Step | Who | Where |
|---|---|---|
| Write and change code, job scripts, configs | AI assistant (you review) | Laptop |
| Test code locally (small inputs, a fake model, a syntax check) | AI assistant | Laptop |
| `git commit` | AI assistant, when you ask | Laptop |
| `git push` | **You** | Laptop |
| Connect to the cluster (SSH or VS Code Remote-SSH) | **You** | Laptop → cluster |
| `git pull`, `sbatch`, `squeue`, `scancel`, reading logs | **You only** | Cluster terminal |
| Fill in cluster-specific values (partition, modules, paths) | **You only** | Cluster (in a gitignored file) |
| Bring results back to the laptop | **You** | VS Code download, or `scp`/`rsync` run by you |
| Analyse results, fix bugs | AI assistant | Laptop |

### Ground rules

- **Never give the AI:** passwords, SSH keys, tokens, hostnames, IP addresses,
  or output from `sinfo`, `hostname` or `squeue` that shows node names.
- **Crop screenshots** that show the cluster's address (window title bars,
  status bars, terminal prompts).
- **When a job fails,** paste only the relevant error lines, with hostnames and
  full paths removed.
- **Read every AI-generated command** before you run it on the cluster.
- **Enforce it technically as well.** In Claude Code, a project
  `.claude/settings.json` can deny the assistant `ssh`, `scp`, `rsync`,
  `sshfs`, and reads of `~/.ssh`:

  ```json
  {
    "permissions": {
      "deny": ["Bash(ssh *)", "Bash(scp *)", "Bash(sftp *)", "Bash(rsync *)",
               "Bash(sshfs *)", "Read(~/.ssh/**)", "Edit(~/.ssh/**)"]
    }
  }
  ```

  With git over SSH, the assistant's shell also can't use your key to push.
  That's why `git push` is always yours, and it's a good thing.

### Keeping cluster details out of git

Put cluster-specific settings in a file that exists **only on the cluster**
and is listed in `.gitignore`. Commit an example version with placeholders:

```bash
# scripts/hpc/cluster.env.example   (committed)
# Copy to cluster.env on the HPC and fill in. cluster.env is gitignored.
# module load <python module name>
export HF_HOME=/scratch/$USER/hf_cache
export DATA_DIR=/scratch/$USER/datasets
```

```gitignore
# .gitignore
scripts/hpc/cluster.env
```

The setup script and every job script `source` this file. The code in git
stays generic, `git pull` never overwrites your settings, and cluster details
never reach GitHub or the AI.

The partition name goes on the command line instead of in the script
(`sbatch -p <partition> job.slurm`), for the same reason.

---

## 3. One-time setup

### 3.1 Connect

Use SSH or **VS Code Remote-SSH**. Remote-SSH gives you a file explorer, an
editor and a terminal on the cluster in one window. Follow your university's
instructions for access (VPN, keys, MFA), and do this yourself.

### 3.2 Get the code onto the cluster

```bash
cd ~
git clone <your repository URL>
cd <repository>
```

Cloning over HTTPS is enough if the cluster only needs to **pull**. You don't
need to put GitHub credentials on the cluster unless you want to push from it.

### 3.3 Fill in cluster settings

```bash
cp scripts/hpc/cluster.env.example scripts/hpc/cluster.env
module avail python          # list the Python versions available
```

Open `cluster.env`, set the `module load` line to a suitable Python version,
and point the large-download paths at scratch. Test it:

```bash
source scripts/hpc/cluster.env && python3 --version && echo "$HF_HOME"
```

### 3.4 Build the software environment (on the login node)

Most clusters offer two routes. Check which one your documentation recommends.

**Option A: a Python virtual environment.** Simple, and usually enough for PyTorch.

```bash
source scripts/hpc/cluster.env
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir --upgrade pip
.venv/bin/pip install --no-cache-dir -r requirements.txt
```

- `--no-cache-dir` stops pip keeping a second multi-GB copy of PyTorch in your home directory.
- PyTorch's Linux wheels come with their own CUDA libraries. The GPU node only needs the NVIDIA driver, which the cluster provides.
- **Pin versions** in `requirements.txt`, so the laptop and the cluster run the same code.

**Option B: a container (Apptainer / Singularity).** Some clusters recommend
this for GPU frameworks. You run your code inside a pre-built image, such as an
official PyTorch container, typically with `module load apptainer` and
`apptainer exec --nv <image> python ...`. The `--nv` flag passes the GPU
through. Use this if Option A runs into driver or library errors.

### 3.5 Download big files before any job runs

**Compute nodes often have no internet access.** Download model weights and
datasets on the login node, into scratch:

```bash
source scripts/hpc/cluster.env           # sets HF_HOME to scratch
.venv/bin/hf download <org/model-name>
```

Downloading is fine on the login node; computing is not. If a download stops
halfway, run it again and it resumes.

**Access tokens** (e.g. Hugging Face): create a **read-only** token and enter
it **yourself** in the cluster terminal, never in a chat with the AI:

```bash
source scripts/hpc/cluster.env           # so the token is stored under $HF_HOME
.venv/bin/hf auth login                  # paste the token; answer "n" to git credential
chmod 600 "$HF_HOME/token"
```

---

## 4. Writing a GPU job script

A generic template. Save it as e.g. `scripts/hpc/my_job.slurm`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my-gpu-job           # name shown in squeue and log names
#SBATCH --nodes=1                       # some clusters make this mandatory
#SBATCH --gres=gpu:1                    # number of GPUs; without this you get NONE
#SBATCH --cpus-per-task=4               # CPU cores (data loading, preprocessing)
#SBATCH --mem=48G                       # system RAM (not GPU memory)
#SBATCH --time=02:00:00                 # hard limit hh:mm:ss; the job is killed after this
#SBATCH --output=logs/%x_%j.out         # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.out          # send errors to the same file

set -euo pipefail                       # stop at the first error

# Cluster settings (modules, paths) from the gitignored file
if [[ -f scripts/hpc/cluster.env ]]; then
  source scripts/hpc/cluster.env
fi
export HF_HUB_OFFLINE=1                 # compute nodes: use cached files, don't try the internet
export PYTHONUNBUFFERED=1               # print() lines reach the log immediately

echo "job $SLURM_JOB_ID started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Fail fast if there's no GPU, rather than silently crawling on the CPU
.venv/bin/python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else "no GPU visible")'

mkdir -p outputs
.venv/bin/python my_script.py --out outputs/result.jsonl

echo "finished $(date)"
```

### What the directives mean

| Directive | Meaning | Tips |
|---|---|---|
| `--gres=gpu:N` | Number of GPUs | **Being in a GPU partition does not give you a GPU.** You must request one. |
| `--nodes=1` | Number of machines | Single-GPU work always uses 1. Some clusters reject jobs that leave it out. |
| `--mem` | System RAM | Loading a model briefly needs about its size in RAM. |
| `--time` | Wall-clock limit | Shorter jobs often start sooner. Estimate, then add about 50%. |
| `--output` | Log file | Create `logs/` first, or keep a `logs/.gitkeep` in the repo. |
| `-p` / `--partition` | Which group of machines | Pass it on the command line: `sbatch -p <partition> ...` |

**Choosing a partition:** list them with `sinfo -s`, and read it yourself, not
with the AI. Check your cluster's limits page for which partitions you may use
and what GPU memory each offers. Rough GPU memory needs in bf16/fp16:

| Model size | Weights alone | GPU memory to aim for |
|---|---|---|
| 1–4B parameters | 2–8 GB | 16 GB+ |
| 7–8B | ~16–18 GB | 24 GB is tight; 40 GB+ is comfortable |
| 30B+ | 60 GB+ | 80 GB, or several GPUs |

Partitions named `*dev*` are usually for short tests and start quickly.
Partitions named `*scavenger*` (or similar) are cheap, but **your job can be
killed at any time** to make room for others.

---

## 5. Submitting and monitoring

```bash
sbatch -p <partition> scripts/hpc/my_job.slurm     # -> "Submitted batch job 123456"
```

| Command | What it shows / does |
|---|---|
| `squeue -u $USER` | Your jobs. **ST**: `PD` = waiting, `R` = running. A job that's missing has finished. |
| `squeue -u $USER --start` | Estimated start times (rough, sometimes `N/A`) |
| `watch -n 10 squeue -u $USER` | Refreshes every 10 s (`Ctrl+C` to stop) |
| `tail -f logs/<job-name>_<jobid>.out` | Live log (`Ctrl+C` stops the viewer, not the job) |
| `sacct -j <jobid> --format=JobID,State,Elapsed` | Final status: `COMPLETED`, `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED` |
| `scancel <jobid>` | Cancel a job (you can only cancel your own) |
| `ls logs/` | Find log names. Or type the start of a name and press **Tab**. |

**Why is my job pending?** The reason in brackets at the end of the `squeue`
line tells you:

| Reason | Meaning |
|---|---|
| `(Resources)` | Waiting for a free GPU. You're near the front. |
| `(Priority)` | Other jobs are ahead of you. |
| `(QOSMax...)` / `(AssocMax...)` | You've hit a per-user limit. Wait for your other jobs. |
| `(ReqNodeNotAvail)` / `(PartitionDown)` | Maintenance or an unavailable partition. Try another one. |

**Interactive sessions** give you a live shell on a GPU node, which is useful
for debugging (e.g. `srun -p <partition> --gres=gpu:1 --pty bash`, or your
cluster's own wrapper). The GPU stays booked while you sit idle, so exit as
soon as you're done, and use batch jobs for real runs.

---

## 6. The development loop

```
 1. AI edits code locally, tests it on tiny inputs  ->  commit
 2. You: git push                                     (laptop)
 3. You: git pull                                     (cluster)
 4. You: sbatch a SMOKE TEST (tiny input, 15 min)     (cluster)
 5. Passed? -> sbatch the FULL job
    Failed? -> paste the error lines (no hostnames) to the AI -> back to 1
 6. You: download results -> AI analyses them locally
```

**Always run a smoke test first:** the same script on 1–2 inputs with a short
time limit. It catches environment, path and memory problems in minutes rather
than after hours in the queue. Have it end with an unmistakable line like
`echo "SMOKE TEST PASSED"`.

---

## 7. Long runs

### Make jobs resumable

Write results as you go (e.g. one JSON line per item, flushed each time), and
on start-up skip items already in the output file. If a job is killed by its
time limit or preempted, submitting it again continues where it stopped.

### Split the work with a job array

One submission, N independent jobs running in parallel, each taking a share:

```bash
#SBATCH --array=0-3                     # 4 tasks: IDs 0,1,2,3
#SBATCH --output=logs/%x_%A_%a.out      # %A = array job ID, %a = task ID

python my_script.py --shard "$SLURM_ARRAY_TASK_ID/4" --out "outputs/part_${SLURM_ARRAY_TASK_ID}.jsonl"
```

Afterwards, combine the parts: `cat outputs/part_*.jsonl > outputs/all.jsonl`.
Each task needs its own GPU, so on a busy cluster they may start at different
times.

### Estimating run time

Measure rather than guess: time a few items in the smoke test.
**Total ≈ items × seconds per item.** For text generation, that's roughly
answer tokens ÷ tokens per second. Add a margin, then set `--time`.

---

## 8. Getting results back

- **VS Code Remote-SSH:** right-click a file or folder in the Explorer → **Download**.
- **Command line (run by you, on the laptop):** `scp` or `rsync` from the cluster.
- **Keep large outputs out of git.** Add `outputs/` and `logs/*` to
  `.gitignore`. Commit code and small configs, not results.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `sbatch: error: ... include #SBATCH --nodes=` | The cluster requires `--nodes` | Add `#SBATCH --nodes=1` |
| `Invalid partition` / `Invalid account or account/partition combination` | You can't use that partition | Pick another one from your cluster's limits page |
| Job runs but is extremely slow; `torch.cuda.is_available()` is False | No GPU requested, or a CPU partition | Add `--gres=gpu:1` and use a GPU partition |
| `CUDA out of memory` | The GPU is too small for the model or batch | Bigger-memory GPU, smaller batch, or lower precision |
| `libcuda.so.1: cannot open shared object file` | Running CUDA code on the login node, or in a container without `--nv` | Run inside a GPU job; add `--nv` for Apptainer |
| `DUE TO TIME LIMIT` in the log / `TIMEOUT` in sacct | `--time` was too short | Raise it, or make the job resumable and resubmit |
| `Disk quota exceeded` | Home is full | Move caches (`HF_HOME`, datasets, pip cache) to scratch |
| `module: command not found` / wrong Python | The module isn't loaded inside the job | `source` your `cluster.env` at the top of the job script |
| Model tries to download inside the job and fails | No internet on compute nodes | Download on the login node first; set `HF_HUB_OFFLINE=1` in the job |
| Two copies of a job overwrite each other's files | Submitted twice | `scancel` the duplicate. If both ran, discard the results and resubmit once. |
| Log stops after model loading for many minutes, yet the job is `R` | Python buffers `print` output written to a file (in ~8 KB chunks) | Check the output file instead. For live logs, add `export PYTHONUNBUFFERED=1` to the job script (or use `python -u`). |

---

## 10. Etiquette

- No heavy computing on the login node. Installing and downloading are fine.
- Request what you need, not the maximum. Over-asking makes your jobs wait longer and blocks others.
- Cancel jobs you no longer need. Don't leave interactive sessions idle.
- Clean up scratch from time to time. It isn't backed up, and some clusters purge old files automatically.
- If in doubt, ask your HPC support team. They'd rather answer a question than clean up after a mistake.

---

## 11. Cheat sheet

```bash
# --- laptop ---
git push                                           # after the AI commits

# --- cluster (you only) ---
cd ~/<repo> && git pull
source scripts/hpc/cluster.env                     # when running things by hand
sbatch -p <partition> scripts/hpc/<job>.slurm     # submit
squeue -u $USER                                    # PD waiting / R running / gone = done
tail -f logs/<name>_<jobid>.out                    # live log
sacct -j <jobid> --format=JobID,State,Elapsed      # did it succeed?
scancel <jobid>                                    # cancel
sinfo -s                                           # partitions (don't share the output)
```
