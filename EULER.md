# Remote machine: the ETH Euler cluster

Status (7 Nov 2026): access works through the user's SSH agent; the environment is installed in main; the test
suite of the merge branch passes on Euler (1755 of 1755 after WP8, job 20155563); main equals the local worktree byte for byte.

- **Purpose.** How to reach Euler, where files go there, how to run a job, and the workflow for testing the package
  `fleet_management` with a real Gurobi licence. For the user and for Claude, in any later session.
- **Euler** is the high-performance computing cluster of ETH Zurich. Jobs run on compute nodes through the job
  scheduler Slurm (Simple Linux Utility for Resource Management); the login node is only for copying files,
  editing and submitting jobs.

## Access

| Item | Value |
|---|---|
| Command | `ssh lfiaschi@euler.ethz.ch` |
| Host keys | `~/.ssh/known_hosts` of the workstation holds the ED25519 and RSA keys of `euler.ethz.ch` |
| Login methods the server offers | public key, and keyboard-interactive (the password) |
| Key of the workstation | `~/.ssh/id_ed25519` (ED25519, fingerprint `SHA256:6UnUq7F03BRwxkCWGeut4VqtuFCdZyxh6Uwkxn9y8ck`), protected by a passphrase |

- **Without the agent** (checked 7 Nov 2026), a login without a terminal (`ssh -o BatchMode=yes ...`) fails with
  `Permission denied (publickey,keyboard-interactive)`: the key needs its passphrase. Only the user types the
  password and the passphrase; Claude never enters them.
- **With the agent** (7 Nov 2026): the login lands on the login node `eu-login-42`, in `/cluster/home/lfiaschi`.
- **The commands below run in Git Bash**, not in PowerShell: PowerShell has no `ssh-copy-id`, and the agent socket
  must be readable from Git Bash. From a PowerShell terminal, start Git Bash with
  `& "C:\Program Files\Git\bin\bash.exe" --login` (a plain `bash` may start the Linux subsystem WSL instead).
- **How Claude logs in** (user's choice, 7 Nov 2026: the SSH agent; tested on the workstation the same day):
  1. Once: the user authorises the workstation key on Euler, in a Git Bash terminal:
     `ssh-copy-id -i ~/.ssh/id_ed25519.pub lfiaschi@euler.ethz.ch` (it asks for the Euler password).
  2. Each working session: the user starts an SSH agent (a background program that holds the unlocked key in memory)
     on a fixed socket, and loads the key:
     `ssh-agent -a ~/.ssh/euler-agent.sock` then `SSH_AUTH_SOCK=~/.ssh/euler-agent.sock ssh-add ~/.ssh/id_ed25519`
     (it asks for the passphrase).
  3. Claude's commands then use the agent: `SSH_AUTH_SOCK=~/.ssh/euler-agent.sock ssh lfiaschi@euler.ethz.ch '<command>'`.
  - The socket path must be short: a Unix socket path in the session scratchpad was refused as too long.
  - The agent runs until it is stopped (`SSH_AUTH_SOCK=~/.ssh/euler-agent.sock ssh-agent -k`) or the user logs out
    of Windows. The private key stays encrypted on disk.

## Folders on Euler

| Folder | Path | Quota (`lquota`, 7 Nov 2026) | Use |
|---|---|---|---|
| main | `/cluster/home/lfiaschi` (the home folder, where a login lands) | 45 GB, 450,000 files (soft); backed up | Keeps the current version of every file: the environment scripts, the source, the tests, the user's example scripts and public key |
| scratch | `/cluster/scratch/lfiaschi` (`../../scratch/lfiaschi` from main) | 2.5 TB, 1 million files (soft) | Where jobs run: copy the files of interest here, then submit from here |

**Rules of scratch** (file `__USAGE_RULES__` in the scratch folder, read 7 Nov 2026):
- files older than 15 days are deleted automatically, without notice;
- nothing is backed up;
- changing time stamps to avoid the deletion is forbidden;
- remove files that are no longer needed;
- many small files run better in home or on a local scratch.

So a file that must last (the code, the environment scripts, the results worth keeping) lives in main; scratch holds
only copies for a run. The user notes (7 Nov 2026) that the 15-day deletion has never happened to them so far; the
rule stays the safe assumption.

## Software modules and Gurobi

Euler loads software through environment modules (Lmod). The software stack in use is `stack/2024-06` (the file
`~/.software_stack_default` says `new`).

| Order | Command | Effect (checked 7 Nov 2026) |
|---|---|---|
| 1 | `module load stack/2024-06 gcc/12.2.0` | the stack and the compiler; must come first |
| 2 | `module load python/3.12.8` | `python3` becomes Python 3.12.8 (without step 1 it stays the system Python 3.10.12) |
| 3 | `module load stack/2024-06 gurobi/13.0.0` | sets `GUROBI_HOME`, `GUROBI_ROOT` and `GRB_LICENSE_FILE` |

- **Licence:** `GRB_LICENSE_FILE` points to a `gurobi.lic` of type `TOKENSERVER`, a licence server of ETH that hands
  out tokens to running programs.
- **gurobipy is not in the module's Python**: `import gurobipy` fails after the three loads. The environment
  installs it from PyPI (the Python Package Index); versions 13.0.0 to 13.0.3 are available (7 Nov 2026). The
  version should match the module (13.0.0).
- Other Gurobi modules: 9.0.0, 10.0.3, 11.0.3, 12.0.1, 13.0.0 (the default).
- **Internet from compute nodes:** the module `eth_proxy` exists; a job that downloads (for example `pip install`)
  loads it. The login node reaches PyPI directly.
- Many modules are hidden in the stack; `module --show_hidden spider <name>` finds them.

## The user's example scripts (in main, read 7 Nov 2026)

`install.sh`, `run.sh` and `uninstall.sh` come from earlier runs. They share one Slurm header:

| Line | Meaning |
|---|---|
| `#!/bin/bash` | run with bash |
| `#SBATCH -n 8` | 8 tasks (Slurm processes); a single Python process uses one task, so `--ntasks=1 --cpus-per-task=8` gives its threads 8 CPUs instead |
| `#SBATCH --time=1-0` | at most 1 day (`days-hours`) |
| `#SBATCH --mem-per-cpu=16G` | 16 GB per CPU, 128 GB in all |
| `#SBATCH --job-name=av-ns-6` | job name (copied from another project) |
| `#SBATCH --output=logs/out-ns-6.txt`, `--error=logs/err-ns-6.txt` | output and error files; the folder `logs/` must exist before `sbatch`, or the job cannot write them |

Their bodies:

- **`install.sh`:** loads `stack/2024-06 gcc/12.2.0` and `python/3.12.8`; activates the virtual environment
  `environment/` (a folder of Python packages, created beforehand with `python3 -m venv environment`; the script
  does not create it); runs `pip install .`, which installs the package of the current folder.
- **`run.sh`:** the same loads plus `stack/2024-06 gurobi/13.0.0`; activates `./venv`; runs
  `remote_smoke_test.py` with the Python of `/cluster/scratch/lfiaschi/fleet_v1/fleet_management/venv`.
- **`uninstall.sh`:** the same loads as `install.sh`; activates `environment/`; runs
  `pip uninstall fleet-management`.

Points to fix in the new scripts:
- the environment is called `environment/` in two scripts and `venv` in the third, and `run.sh` uses an absolute
  path in scratch that no longer exists (scratch was empty on 7 Nov 2026);
- `install.sh` does not create the environment, and does not load `eth_proxy` for `pip` on a compute node;
- `pip uninstall` without `-y` asks for confirmation, which a batch job cannot give;
- the job name and log names belong to another project;
- 8 tasks of 16 GB for one day are far more than an install needs, and a large request waits longer in the queue.

## Running a job with Slurm

Source: the Euler tutorial "Your first job", https://docs.hpc.ethz.ch/tutorials/first-job/ (read in full,
7 Nov 2026).

**A job script** is a shell script whose first lines are `#SBATCH` options:

| Option | Example | Meaning |
|---|---|---|
| `--job-name` | `my_first_job` | Name shown in the queue |
| `--time` | `00:05:00` | Maximum run time; the job is stopped after it |
| `--ntasks` | `1` | Number of tasks (1 for a serial job) |
| `--cpus-per-task` | `1` | CPUs per task |
| `--mem-per-cpu` | `1G` | Memory per CPU |
| `--output` | `first_job_%j.out` | File for standard output; `%j` becomes the job ID |
| `--error` | `first_job_%j.err` | File for standard error |

Inside the script, `$SLURM_JOB_ID`, `$SLURMD_NODENAME` and `$SLURM_CPUS_PER_TASK` give the job ID, the node and
the CPUs; arguments after the file name on the `sbatch` line arrive as `$1`, `$2`, ...

**Commands**

| Command | What it does |
|---|---|
| `sbatch <file>` | Submits the job; prints `Submitted batch job <JOBID>` |
| `squeue -j <JOBID>`, `squeue -u $USER`, `myjobs` | State of one job, or of all the user's jobs |
| `scancel <JOBID>` | Cancels a job |
| `sacct --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS` | History of finished jobs, with exit code and memory used |
| `srun --time=00:10:00 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=1G --pty bash` | Interactive shell on a compute node; `exit` ends it |
| `module avail` | Lists the software modules |
| `lquota` | Storage quotas of home, scratch and group storage |

**Job states:** `PD` pending (waiting in the queue), `R` running, `CG` completing, `CD` completed.

**Common problems**

| Symptom | What to check |
|---|---|
| Stays in `PD` | The REASON column of `squeue -j <JOBID>`: too many resources requested, a busy cluster, or an invalid option |
| Fails at once | The `.err` file, `sacct -j <JOBID> --format=ExitCode`, and that the script is executable (`chmod +x`) |
| "Permission denied" | `chmod +x <file>` |
| Killed for memory | Raise `--mem-per-cpu` |

## Workflow for testing the package (user's instructions, 7 Nov 2026)

1. A Python environment on Euler, with the source and the test files of `fleet_management`: the package of the
   branch `merge/lore-into-main` and its pytest suite `tests/` (user's choice, 7 Nov 2026; `lore`'s old `test/`
   folder stays out until WP8 ports it).
2. Three shell scripts: one installs the environment, one removes it, one runs the tests.
3. Copy them, with the source and the tests, into main; copy the files needed for a run into scratch.
4. From scratch: install the environment, then launch the tests with `sbatch`.
5. Check the results, change the code, and repeat until the tests pass.
6. Leave the final version of the code in main, and download it back to the workstation.

**Who does what.** Sonnet agents do the work: writing the scripts, copying files, editing the code. The main
session plans, submits the jobs (`sbatch`), and checks the results.

**Choices (user, 7 Nov 2026)**

| Item | Choice |
|---|---|
| Virtual environment | `/cluster/home/lfiaschi/fleet_management/venv`, in main (backed up, never purged); it may move to scratch if something goes wrong there |
| `gurobipy` | `13.0.0`, the version of the module `gurobi/13.0.0` |
| Uninstall | removes the package only (`pip uninstall -y fleet-management`); an outdated environment (Python not 3.12.8, `gurobipy` not 13.0.0, or its base Python gone) is also removed, by a move to `/cluster/scratch/lfiaschi/trash/` (the user's rule: no hard delete) |
| Scripts in the repository | `euler/` of the merge branch, next to `main`'s two Slurm jobs |

**Layout**

| Place | Content |
|---|---|
| main: `~/fleet_management/` | `src/`, `tests/`, `pyproject.toml`, `results/validator_tests/` (`input.yaml`, `result.yaml`: fixtures that the validator tests read), `env_install.sh`, `env_uninstall.sh`, `run_tests.sh`, `venv/` |
| scratch: `/cluster/scratch/lfiaschi/fleet_management/` | a copy of the same files except `venv/`, plus `logs/`; jobs are submitted from here |
| workstation | the merge worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`; the scripts in `euler/` |

The tests run on the scratch copy of the code (`PYTHONPATH` points to its `src/`); the environment supplies the
dependencies.

## Runs

| Date | Job | What | Result |
|---|---|---|---|
| 7 Nov 2026 | 20142049 | `env_install.sh` | completed in 1 min 25 s; Python 3.12.8, gurobipy 13.0.0, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, pytest 9.2.0 |
| 7 Nov 2026 | 20142478 | `run_tests.sh` | licence token from `lic-gurobi.ethz.ch`; 514 tests, 464 passed, 50 failed in 37 s; all 50 failures in `test_validator.py` and `test_validator_dashboards.py`, because `results/validator_tests/` was not copied |
| 7 Nov 2026 | 20143040 | `run_tests.sh`, after copying the two fixtures | 514 passed in 34 s, no warnings; the three `ERROR` lines in the log are captured logs of sweep tests that inject a failure ("boom 5") on purpose; code commit `7e35db4`, scripts commit `fb6097a` |
| 7 Nov 2026 | 20153161 | `run_tests.sh`, WP8 suite without the brute-force and ported files | 1597 passed, 8 failed: one test that expected the wrong error, and an infeasible test input (both test defects, fixed) |
| 7 Nov 2026 | 20153605 | `run_tests.sh tests/test_oracle_bruteforce.py` | 85 passed in 63 s |
| 7 Nov 2026 | 20153607 | `run_smoke.sh --no-pytest` (new smoke test, 12 scenarios) | 12 passed in 5.6 min |
| 7 Nov 2026 | 20154950 | `run_tests.sh`, whole WP8 suite | 1754 passed in 226 s |
| 7 Nov 2026 | 20154951 | `run_smoke.sh --no-pytest` | 12 passed |
| 7 Nov 2026 | 20154956 | `run_regression.sh` (six check scripts of `examples/regression/`; the script was copied from the session scratchpad) | 6 passed in 46 s |
| 7 Nov 2026 | 20155563 | `run_tests.sh`, after the last fixes (parallel workers with `spawn`) | 1755 passed in 226 s; one expected warning; WP8 commit `8cf16c1`; main equals the worktree (196 files, checksums) |

## One test cycle, step by step

1. **Workstation:** start the SSH agent if needed (*Access*).
2. **Copy to main** (a Sonnet agent): `src/`, `tests/`, `pyproject.toml`, `results/validator_tests/input.yaml` and
   `result.yaml`, and the three scripts of `euler/`, into `~/fleet_management/`; no `__pycache__`, `*.pyc` or
   `*.egg-info`.
3. **Copy to scratch** (a Sonnet agent): `cp -r` the same files from `~/fleet_management/` to
   `/cluster/scratch/lfiaschi/fleet_management/`, without `venv/`, and `mkdir -p logs`.
4. **Environment** (main session; only when it is missing or outdated): from the scratch folder,
   `sbatch env_install.sh`.
5. **Tests** (main session): from the scratch folder, `sbatch run_tests.sh` (the whole suite), or
   `sbatch run_tests.sh tests/test_x.py` (only the given pytest targets); wait with `squeue -h -j <JOBID>`;
   then `sacct -j <JOBID>` and the files `logs/tests_<JOBID>.out` and `.xml` (copy them to the workstation with
   `scp` to group the failures by message).
6. **Fix and repeat** (a Sonnet agent edits the code in the local worktree; steps 2 and 3 copy it again).
7. **End:** compare checksums of main and the local worktree on raw bytes (`md5sum` on both sides, without
   changing line endings); equal files mean the final version is the same in both places.

- The scripts need Unix line endings; `.gitattributes` of the merge branch keeps them for `*.sh` and `*.sbatch`.
- In Git Bash on the workstation, `grep -c $'\r'` counts lines instead of carriage returns; count them with
  `tr -cd '\r' < <file> | wc -c`.

## Copying files and running commands from the workstation

- Run one command on Euler: `ssh lfiaschi@euler.ethz.ch '<command>'` (with the agent of *Access*).
- Copy to Euler: `scp -r <local path> lfiaschi@euler.ethz.ch:<remote path>`.
- Copy back: `scp -r lfiaschi@euler.ethz.ch:<remote path> <local path>`.

## Open points

- The package's own input examples: `lore`'s `input/` folder is not yet on the workstation (`test/` arrived on
  2 Nov 2026).
