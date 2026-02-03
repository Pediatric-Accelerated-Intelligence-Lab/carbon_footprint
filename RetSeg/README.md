# RetSeg branch — Federated 2D Segmentation (FedAvg + CodeCarbon)

This branch runs federated 2D segmentation across multiple sites using a simple FedAvg loop built with PyTorch Lightning, with per-site carbon tracking via CodeCarbon.

Main entry points:
- train.py — federated training (FedAvg)
- infer_fedavg.py — evaluate a saved global checkpoint on each site

Core modules:
- data.py — site-aware data module (JSON-driven) + MONAI transforms
- learner.py — LightningModule wrapper for segmentation + Dice metrics
- unet.py — MONAI-style UNet with dynamic active depth (used by learner.py)
- learner_multiexits.py — optional multi-exit variant (not used by default)
- seed.py — reproducibility utilities

Install:
pip install -r requirements.txt

Dataset layout (required):
By default, the dataloader expects a dataset root at:
- /Data/
- Site split JSON files: /Data/dataset_site1.json, /Data/dataset_site2.json, ..., /Data/dataset_site5.json

Each dataset_site*.json must contain three keys: training, validation, testing.
Each is a list of records with at least:
{"image": "relative/path/to/image.png", "label": "relative/path/to/label.png"}

The loader will prepend /Data/ to image and label.
If your dataset is not under /Data, edit dir_path in SiteSegDataModule2D inside data.py.

Run federated training (FedAvg):
python train.py

Defaults:
- 5 sites: [1, 2, 3, 4, 5]
- 30 federated rounds
- 10 local epochs per round per site
- Image/patch size: 256
- Batch size per site: 96
- Mixed precision: 16-mixed
- GPU: script sets CUDA_VISIBLE_DEVICES=0 and Lightning uses accelerator="gpu", devices=[0]

Outputs (default ./Output/):
- TensorBoard logs: ./Output/tb_logs_fedavg/
- Local checkpoints: ./Output/saved_models_fedavg/local_federated_fedavg_site*/
- Global checkpoints: ./Output/saved_models_fedavg/global_federated_fedavg_round{R}.pth

Carbon tracking:
Training wraps each site’s local fit in a CodeCarbon.EmissionsTracker(...). This produces CodeCarbon’s standard emissions artifacts (location/configurable depending on your CodeCarbon settings).

Evaluate a trained global model (per-site validation):
python infer_fedavg.py

By default it expects:
- global_federated_fedavg_round30.pth in the current working directory

If your checkpoint is saved under ./Output/saved_models_fedavg/, either copy it to the working directory or edit pth_path in infer_fedavg.py to point to the file you want.

Notes on the model / compute settings:
- The default learner (SegLearnerDepthFL2D) uses a UNet that supports dynamic “active depth” via active_layers.
- For FedAvg in this branch, train.py sets compute_capacities = [5] * n_sites and explicitly notes not to change it.
- If you want to explore heterogeneous compute (depth-varying clients), see:
  - unet.py for how bypass paths and learnable-parameter selection are handled
  - learner_multiexits.py for an alternative multi-exit setup

Reproducibility:
Both training and inference seed RNGs and configure deterministic behavior via seed_everything(...).
