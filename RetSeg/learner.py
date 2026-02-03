
import torchvision
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
import monai
from pytorch_lightning import LightningModule


import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric
import numpy as np
from model.unet import UNet

class SegLearnerDepthFL2D(LightningModule):
    def __init__(self, lr=1e-4, img_size=224, compute_capacity=1, state_dict=None):
        super().__init__()

        self.save_hyperparameters(ignore=['state_dict'])
        self.example_input_array = torch.ones(10, 3, self.hparams.img_size,self.hparams.img_size)

        self.model = UNet(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                active_layers=compute_capacity,
                act='RELU',
                num_res_units=2)
        
        flops = self.model.estimate_flops(input_shape=(10, 3, self.hparams.img_size,self.hparams.img_size), active_layers=self.hparams.compute_capacity)
        print(f"active_layers={self.hparams.compute_capacity}: FLOPs={flops/1e9:.3f} GFLOPs")

        if state_dict is not None: 
            print(self.load_state_dict_trainer(state_dict))

        self.loss_function = monai.losses.DiceLoss(sigmoid=True)

        self.dice_calc = DiceMetric(reduction='none')
        

    def load_state_dict_trainer(self, state_dict, strict: bool = True):
        print(self.model.load_state_dict(state_dict, strict=strict))
        
    def forward(self, images):
        return  self.model(images, active_layers=self.hparams.compute_capacity)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, label = self._prepare_batch(batch)
        return self.forward(img), label

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for _, p in self.model.get_learnable_parameters(self.hparams.compute_capacity)],
            lr=self.hparams.lr,
        )
        
        lr_scheduler = {
            'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, cooldown=5),
            'monitor': 'val_total_loss'
        }
        return [optimizer], [lr_scheduler]

    def _prepare_batch(self, batch, stage):
        if stage == 'train':
            batch = batch[0]
        return batch['image'], batch['label']

    def calc_losses(self, images, preds, label, stage):
        # criterion- diceloss
        r_loss = self.loss_function(preds, label)
        self.log(f"{stage}_total_loss", r_loss, on_step=True, batch_size=images.shape[0], sync_dist=True)
        return r_loss

    def _common_step(self, batch, batch_idx, stage: str):
        images, label= self._prepare_batch(batch, stage)
        preds = self.forward(images)
        if stage == 'val':
            preds = (F.sigmoid(preds)>0.5).float()
            dice = self.dice_calc(preds, label)
            self.metrics += list(dice.flatten().cpu().numpy())

            # if self.global_rank==0 and batch_idx <= 1:
            #     img_grid = torchvision.utils.make_grid(images[:40, ...])
            #     pred_grid = torchvision.utils.make_grid(torch.cat([preds[:40, ...],label[:40,...],torch.zeros_like(label[:40,...]) ], 1))
            #     plot_2d_or_3d_image(data=torch.concat([img_grid, pred_grid], axis=-1).unsqueeze(0), max_channels=3, step=self.current_epoch, writer= self.logger.experiment, tag=f"pred_{stage}_{batch_idx}")

        recon_loss = self.calc_losses(images, preds, label, stage)
        return recon_loss

    def on_validation_epoch_start(self):
        self.metrics = []
    
    def on_validation_epoch_end(self):
        self.log(f"dice:", np.mean(self.metrics), batch_size=1 , sync_dist=True)
        return {'dice': np.mean(self.metrics)}
    
    def get_metrics(self):
        learnable_params = self.model.get_learnable_parameters(active_layers=self.hparams.compute_capacity)
        return learnable_params, {'dice': np.mean(self.metrics), 'std': np.std(self.metrics), 'metr': self.metrics}
  