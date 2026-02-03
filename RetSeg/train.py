"""
client side training scripts
"""
# set cuda visible devices
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.seed import seed_everything
seed_everything(1008)

from modules.data import SiteSegDataModule2D
from modules.learner import SegLearnerDepthFL2D
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint#, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


import torch
import numpy as np
torch.set_float32_matmul_precision('medium')
from codecarbon import EmissionsTracker
import pandas as pd

def main():
    rounds = 30 # number of federated rounds
    size = 256 # input image size
    batch_size = 96 # per site batch size

    sites = [1, 2, 3, 4, 5]
    ref_sites = ['A', 'B', 'C', 'D', 'E']
    compute_capacities = [5] * len(sites) # DO NOT CHANGE for FedAvg
    global_epoch_target = [0] * len(sites)
    epochs_per_round_site = [10] * len(sites) #[120, 60, 120, 25, 15]  # local epochs per round per site
    output_dir = "./Output/" # change to your desired output directory
    os.makedirs(output_dir, exist_ok=True)
    logger_global = TensorBoardLogger(os.path.join(output_dir,"tb_logs_fedavg"), name='federated_fedavg_global')
    setup_site = {}
    for i, site in enumerate(sites):
        print(f"Setting up site {site}...")
        name = f'federated_fedavg_site{site}'
        logger = TensorBoardLogger(os.path.join(output_dir,"tb_logs_fedavg"), name=name)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(output_dir,f"saved_models_fedavg/local_{name}/"), save_top_k=1, monitor="val_total_loss", mode='min', save_weights_only=True, save_last=True)
        #early_stopping_callback = EarlyStopping(monitor="val_total_loss", patience=4, min_delta=0.0, mode="min")
        setup_site[site] = {
            'name': name,
            'trainer': Trainer(
                logger= logger,
                accelerator='gpu', 
                devices=[0],
                max_epochs=global_epoch_target[i],
                callbacks=[lr_monitor, checkpoint_callback], # early_stopping_callback],
                # overfit_batches=1,
                log_every_n_steps=1,
                #strategy="ddp_find_unused_parameters_false",
                precision="16-mixed",
            ),
            'model': SegLearnerDepthFL2D(lr=1e-3, img_size=size, compute_capacity=compute_capacities[i]),
            'datamodule': SiteSegDataModule2D(
                batch_size=batch_size,
                patch_size=size,
                using_dataset_sites=[str(sites[i])]),
        }
    print("Setup complete.")

    print("Starting FL training...")
    # get largest site model as initial global model
    testing_site = sites[np.argmax(compute_capacities)]
    main_state_dict = setup_site[testing_site]["model"].state_dict()
    main_state_dict = {k.replace('model.model.', 'model.'):v for k,v in main_state_dict.items() if 'model.model.' in k}
    print("Initial global model prepared.")
    print(main_state_dict.keys())

    for r in range(rounds):
        print(f"Starting round {r+1}/{rounds}...")
        for i, site in enumerate(sites):
            global_epoch_target[i] += epochs_per_round_site[i]
        local_weights = {}
        all_keys = set()

        for site in sites:
            with EmissionsTracker(project_name=f"train_model_site_{site}") as tracker:
                trainer = setup_site[site]["trainer"]
                model   = setup_site[site]["model"]

                trainer.fit_loop.max_epochs = global_epoch_target[sites.index(site)]
                print(f"Loading global model to site {site} for round {r+1}...")
                model.load_state_dict_trainer(main_state_dict, strict=False)

                trainer.fit(model, datamodule=setup_site[site]["datamodule"])
                print(f"Site {site} finished.")
                
            learnable_keys, _ = model.get_metrics()

            local_weights[site] = {
                k: v for k, v in learnable_keys
            }
            all_keys.update(local_weights[site].keys())
        

        main_state_dict = {k:v for k,v in main_state_dict.items() if k not in all_keys}
        # Simple averaging aggregation
        print("Aggregating global model...")
        for key in all_keys:
            tensors = []
            for site in sites:
                if key in local_weights[site].keys():
                    t = local_weights[site][key]

                    # normalize for safe aggregation
                    if torch.is_tensor(t):
                        #print(f"Site {site} contributing {key} of shape {tuple(t.shape)}")
                        tensors.append(t.detach().float().cpu())

            if tensors:
                main_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)

        testing_site = sites[np.argmax(compute_capacities)]
        trainer = setup_site[testing_site]["trainer"]
        model   = setup_site[testing_site]["model"]
        print(main_state_dict.keys())
        model.load_state_dict_trainer(main_state_dict, strict=False)
        for site in sites:
            print(f"Testing global model on site {site}...")
            datamodule = setup_site[site]["datamodule"]
            trainer.validate(model, datamodule=datamodule)
            _, metrics = model.get_metrics()
            logger_global.log_metrics({f"site{site}_dice": metrics['dice']}, step=r+1)
            logger_global.log_metrics({f"site{site}_std": metrics['std']}, step=r+1)

        # save global model
        torch.save(main_state_dict, os.path.join(output_dir,
                   f"saved_models_fedavg/global_federated_fedavg_round{r+1}.pth"))
        print(f"Round {r+1}/{rounds} complete.")

if __name__ == "__main__":
    main()


