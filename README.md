# Standardized Methods for Green Federated Learning (Carbon Accounting)

This repository accompanies the paper **“Standardized Methods and Recommendations for Green Federated Learning”** and provides reference implementations for *phase-aware* carbon accounting in federated learning (FL): compute + coordination/idle + communication.

The code is organized into two experiment branches:
- **`CIFAR`** — CIFAR-10 image classification in **NVFlare** simulation with a FedAvg controller that aggregates per-client CodeCarbon logs and estimates communication emissions.
- **`RetSeg`** — Retinal optic disk segmentation with a lightweight, standalone FedAvg-style loop (PyTorch Lightning) instrumented with **CodeCarbon**.

> **Start here:** pick the folder that you would like to use:
> - `./CIFAR`
> - `./RetSeg`

---

## What this repo is for

Federated learning studies often report emissions inconsistently. This repo provides a *pragmatic, reproducible* measurement boundary and logging schema that separates:
1. **Client compute emissions** (CPU/GPU/RAM)
2. **Client non-training overhead / idle time** (coordination, waiting)
3. **Communication emissions** estimated from transmitted model-update sizes under a configurable network energy model

The goal is to make “green FL” results comparable across workloads, sites, and hardware tiers.

---

## Repository layout

### `CIFAR`
- NVFlare job scripts for CIFAR-10 FL simulation (FedAvg)
- Carbon-aware FedAvg controller that collects per-client emissions and estimates communication CO₂e
- “Efficiency tier” variants (e.g., high/medium) to simulate slowdowns/overhead

### `RetSeg`
- Standalone 2D segmentation FL loop across multiple sites
- CodeCarbon instrumentation around per-site training
- TensorBoard logging + per-round checkpointing of the aggregated global model

---

## Citation

If you use this code, please cite the accompanying paper:

@misc{tapp2026standardized,
  title         = {Standardized Methods and Recommendations for Green Federated Learning},
  author        = {Tapp, Austin and Roth, Holger R. and Xu, Ziyue and Parida, Abhijeet and Nisar, Hareem and Linguraru, Marius George},
  year          = {2026},
  eprint        = {2602.00343},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  doi           = {10.48550/arXiv.2602.00343},
  url           = {https://arxiv.org/abs/2602.00343}
}

