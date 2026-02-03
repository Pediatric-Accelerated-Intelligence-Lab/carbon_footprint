# CIFAR branch — NVFlare CIFAR-10 FedAvg + Carbon Accounting (CodeCarbon)

This branch runs **federated CIFAR-10 classification** using **NVFlare simulation** with a FedAvg controller that:

* aggregates per-client **CodeCarbon** emissions (train + optional idle), and
* estimates **communication energy / CO₂e** from model-update payload size.

It includes three “efficiency tier” run scripts (High/Med/Low) that simulate slowdown/overhead, plus a baseline job script.

---

## What’s in this branch

* `job.py` — baseline NVFlare simulator job (8 clients, 10 rounds; uses `DATASET_PATH`)
* `jobHigh.py` — **FAST** tier (no artificial delay; 6 clients, 10 rounds)
* `jobMed.py` — **MED** tier (adds extra forward-only iterations; 6 clients, 10 rounds)
* `jobLow.py` — **SLOW** tier (extra iterations + per-step sleep; 6 clients, 10 rounds)
* `cifar10_pt_fl.py` — per-client training loop + CodeCarbon tracking + NVFlare client API
* `fedavg_carbon.py` — server/controller: FedAvg + comm-energy/emissions estimate + CSV/PKL export
* `cifar10_data.py` — Dirichlet split generator for heterogeneous client partitions
* `requirements.txt` — Python deps

---

## Requirements

* Python 3.8+
* `nvflare` installed and importable in the active environment
* CUDA-visible GPU(s) if you want GPU training (otherwise adjust NVFlare settings / reduce clients)

---

## Install

```bash
pip install -r requirements.txt
```

---

## Dataset location and splits (IMPORTANT)

### Default expectations

This branch assumes CIFAR-10 lives under **`/data`** and will be downloaded there by the split step if missing.

* CIFAR-10 root: `/data`
* Split indices: `/data/cifar10_splits/`

### If you do NOT have `/data`

Pick one:

1. Create `/data` and make it writable, **OR**
2. Change hard-coded `/data` paths in:

   * `jobHigh.py`
   * `jobMed.py`
   * `jobLow.py`
   * `cifar10_pt_fl.py`

> `job.py` uses `DATASET_PATH` for the split directory, but the dataset itself is still expected under `/data` by default unless you modify the training/data code.

---

## Quickstart (recommended)

### High tier (fast; no slowdown)

```bash
python jobHigh.py
```

### Med tier (extra forward-only iterations; no sleep)

```bash
python jobMed.py
```

### Low tier (extra iterations + per-step sleep)

```bash
python jobLow.py
```

Each tier script typically:

1. partitions CIFAR-10 into per-client splits (Dirichlet alpha=1.0)
2. exports NVFlare configs to `./job_configs`
3. launches an NVFlare simulator run (default `gpu="0"`)

Runs are named:

* `runHigh`
* `runMed`
* `runLow`

**IMPORTANT:** Running multiple tiers back-to-back may overwrite `./job_configs`.
To avoid this, change each tier script to export configs to a unique folder, e.g.:

* `./job_configs_high`
* `./job_configs_med`
* `./job_configs_low`

---

## Baseline job (multi-GPU mapping)

`job.py` runs 8 clients and maps clients across multiple GPUs:

```bash
export DATASET_PATH=/data
python job.py
```

By default it runs:

```python
job.simulator_run("/carbon_footprint", gpu="0,1,2,3")
```

If you only have one GPU:

* change `gpu="0,1,2,3"` → `gpu="0"`
* and/or reduce `n_clients`

---

## Run configs (where they are generated + how to change them)

### Where configs go

All job scripts export NVFlare simulator configs into:

* `./job_configs`

This folder may be overwritten by subsequent runs unless you make it tier-specific.

### What “run configs” include

The generated NVFlare config typically encodes:

* number of clients
* number of rounds
* controller selection (FedAvg + carbon-aware controller)
* client training task settings (script + args)
* server/controller parameters
* dataset/split path information used by clients (directly or indirectly)

### What to edit when changing runs

Open `job.py` / `jobHigh.py` / `jobMed.py` / `jobLow.py` and edit:

* `n_clients` / `num_clients`
* `num_rounds` / `rounds`
* `run_name` (e.g., `runHigh`, `runMed`, `runLow`)
* config output folder (default `./job_configs`)
* simulator call: `job.simulator_run(<workspace>, gpu=<...>)`
* training args passed into `cifar10_pt_fl.py` (tier knobs)

---

## Carbon accounting outputs

After the simulator finishes, the FedAvg controller writes:

* `client_emissions.pkl` — raw per-client per-round emissions metadata
* `client_emissions.csv` — flattened table including:

  **TRAIN (CodeCarbon “train” task):**

  * emissions (kgCO₂e)
  * cpu/gpu/ram energy
  * total energy

  **IDLE (CodeCarbon “idle_time” task) (if enabled/recorded):**

  * emissions (kgCO₂e)
  * cpu/gpu/ram energy
  * duration

  **COMM ESTIMATE:**

  * update size (GB)
  * comm energy (kWh)
  * comm emissions (kgCO₂e)

Where these files are written depends on `fedavg_carbon.py` (commonly either the current working directory or the NVFlare run workspace). To confirm, search `fedavg_carbon.py` for file write calls to:

* `client_emissions.pkl`
* `client_emissions.csv`

---

## Communication emissions model

Per client per round:

**E_comm (kWh) = 2 · D_GB · I_net**
**C_comm (kg CO₂e) = E_comm · F_grid**

Where:

* `D_GB` = model-update payload size in GB
* `I_net` = network intensity (kWh/GB), default `0.01`
* `F_grid` = grid factor (kg/kWh), default `0.475`
* `2` accounts for send + receive

Change these defaults in `fedavg_carbon.py` via:

```python
FedAvg(inet_kwh_per_gb=..., grid_kg_per_kwh=...)
```

---

## CodeCarbon configuration (token vs offline)

### Token / API mode

`cifar10_pt_fl.py` uses something like:

```python
EmissionsTracker(..., api_key=os.getenv("CODECARBON_API_TOKEN"))
```

Set your token:

```bash
export CODECARBON_API_TOKEN="..."
```

### Offline mode (no token)

If you don’t have a token, switch to offline tracking by editing `cifar10_pt_fl.py` to use:

* `OfflineEmissionsTracker`

(The import is commonly already present.)

---

## Efficiency-tier knobs (what High/Med/Low change)

These flags are passed into `cifar10_pt_fl.py` by the job scripts:

* `--extra_no_update_iters`
  Adds extra forward-only passes per batch (no backward/optimizer step).
* `--sleep_ms_mean` / `--sleep_ms_std`
  Adds a per-step Gaussian sleep delay (clipped at 0ms).

Tier defaults (as implemented by the tier scripts):

* **HIGH:** `sleep=0`, `extra_no_update_iters=0`
* **MED:**  `sleep=0`, `extra_no_update_iters=100`
* **LOW:**  `sleep≈500±250ms`, `extra_no_update_iters=100`

---

## Missing data handling (IMPORTANT)

In real runs, you may see **missing/blank/NaN values** in `client_emissions.csv`. This can be expected depending on environment and tracker behavior.

### Common reasons fields are missing

* CodeCarbon tracker did not start/stop cleanly (e.g., exception during training)
* GPU energy is not measurable on the system (GPU energy fields may be blank)
* Idle tracking is not enabled/recorded (idle fields may be blank)
* Token/API issues (if using API mode)
* Container/permission limits preventing energy readings

### Recommended interpretation

* Treat missing values as **unknown (NaN)** by default.
* Only fill missing values with **0** if you explicitly want a conservative *lower-bound* estimate.

### If lots of values are missing

* If you don’t have a token, switch to `OfflineEmissionsTracker`.
* Check client logs to confirm tracker start/stop events.
* Accept that some systems cannot report GPU energy, even if training is correct.

---

## Common customizations

### Change number of clients / rounds

Edit `job*.py`:

* `n_clients` / `num_clients`
* `num_rounds` / `rounds`

### Change which GPU(s) are used

Edit the simulator run call in `job*.py`, e.g.:

* `gpu="0"` (single GPU)
* `gpu="0,1,2,3"` (multi-GPU mapping)

### Change dataset root (recommended if you don’t have `/data`)

Update:

* hard-coded `/data` paths in tier jobs and training code
* `DATASET_PATH` usage in `job.py` (for split directory)

### Prevent run-config overwrites (recommended)

Change each tier job script to export configs to a unique folder instead of `./job_configs`, e.g.:

* `./job_configs_high`
* `./job_configs_med`
* `./job_configs_low`

---

## Troubleshooting

* **DATASET NOT FOUND:**

  * ensure `/data/cifar-10-batches-py` exists, **OR**
  * allow the split step to download CIFAR-10 into `/data`

* **PERMISSION DENIED ON `/data`:**

  * use a writable data directory and update all `/data` references in:
    `jobHigh.py`, `jobMed.py`, `jobLow.py`, `cifar10_pt_fl.py`

* **NO CODECARBON TOKEN:**

  * set `CODECARBON_API_TOKEN`, **OR**
  * switch to `OfflineEmissionsTracker`

* **CONFIGS KEEP GETTING OVERWRITTEN:**

  * change each `job*.py` to export to a unique config folder instead of `./job_configs`
