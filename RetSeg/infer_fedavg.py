"""
client side training scripts
"""
from utils.seed import seed_everything
seed_everything(1008)

from modules.data import SiteSegDataModule2D
from modules.learner import SegLearnerDepthFL2D
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


import torch
import numpy as np
torch.set_float32_matmul_precision('medium')


def main():
    size = 256
    batch_size = 96

    sites = [1, 2, 3, 4, 5]
    ref_sites = ['A', 'B', 'C', 'D', 'E']
    compute_capacities = [5] * len(sites)
    global_epoch_target = [0] * len(sites)
    setup_site = {}
    for i, site in enumerate(sites):
        print(f"Setting up site {site}...")
        name = f'federated_heterofl_site{site}'
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(dirpath=f"./saved_models/local_{name}/", save_top_k=1, monitor="val_total_loss", mode='min', save_weights_only=True, save_last=True)
        #early_stopping_callback = EarlyStopping(monitor="val_total_loss", patience=4, min_delta=0.0, mode="min")
        setup_site[site] = {
            'name': name,
            'trainer': Trainer(
                accelerator='gpu', 
                devices=[0],
                max_epochs= global_epoch_target[0],
                callbacks=[lr_monitor, checkpoint_callback], # early_stopping_callback],
                # overfit_batches=1,
                log_every_n_steps=1,
                #strategy="ddp_find_unused_parameters_false",
                precision="16-mixed",
                logger = False,
            ),
            'model': SegLearnerDepthFL2D(lr=1e-3, img_size=size, compute_capacity=compute_capacities[i]),
            'datamodule': SiteSegDataModule2D(
                batch_size=batch_size,
                patch_size=size,
                using_dataset_sites=[str(sites[i])]),
        }
    print("Setup complete.")

    # get largest site model as initial global model
    pth_path = "global_federated_fedavg_round30.pth" #model path
    main_state_dict = torch.load(pth_path)
    #import pdb; pdb.set_trace()
    #main_state_dict = {k.replace('model.model.', 'model.'):v for k,v in main_state_dict.items() if 'model.model.' in k}
    print("Initial global model prepared.")
    print(main_state_dict.keys())

    
    for site in sites:
        trainer = setup_site[site]["trainer"]
        model   = setup_site[site]["model"]
        model.load_state_dict_trainer(main_state_dict, strict=False)
        print(f"Testing global model on site {site}...")
        datamodule = setup_site[site]["datamodule"]
        trainer.validate(model, datamodule=datamodule)
        _, metrics = model.get_metrics()
        print(f"site{site}_dice",  metrics['dice'], f"site{site}_std", metrics['std'])


if __name__ == "__main__":
    main()
