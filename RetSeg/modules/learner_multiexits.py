import monai
from pytorch_lightning import LightningModule


import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric
import numpy as np
from model.unet import UNetMultiExits

class SegLearnerMultiExitFL2D(LightningModule):
    def __init__(self, lr=1e-4, img_size=224, state_dict=None, compute_capacity=5):
        super().__init__()

        self.save_hyperparameters(ignore=['state_dict'])
        self.example_input_array = torch.ones(10, 3, self.hparams.img_size,self.hparams.img_size)

        self.model = UNetMultiExits(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                act='RELU',
                active_layers=compute_capacity)

        if state_dict is not None: 
            print(self.load_state_dict_trainer(state_dict))

        self.loss_function = monai.losses.DiceLoss(sigmoid=True)

        self.dice_calc = DiceMetric(reduction='none')
        

    def load_state_dict_trainer(self, state_dict, strict: bool = True):
        print(self.model.load_state_dict(state_dict, strict=strict))
        
    def forward(self, images):
        return  self.model(images)

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
            [p for _, p in self.model.named_parameters() if p.requires_grad],
            lr=self.hparams.lr,
        )

        #print([p for _, p in self.model.named_parameters() if p.requires_grad])
        
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
        total_r_loss = 0
        for i, pred in enumerate(preds):
            r_loss = self.loss_function(pred, label)
            self.log(f"{stage}_loss_level_{i+1}", r_loss, on_step=True, batch_size=images.shape[0], sync_dist=True)
            total_r_loss += r_loss

        total_r_loss = total_r_loss / len(preds)
        self.log(f"{stage}_total_loss", total_r_loss, on_step=True, batch_size=images.shape[0], sync_dist=True)
        return total_r_loss

    def _common_step(self, batch, batch_idx, stage: str):
        images, label= self._prepare_batch(batch, stage)
        preds = self.forward(images)
        if stage == 'val':
            for i, pred in enumerate(preds):
                pred = (F.sigmoid(pred)>0.5).float()
                dice_lvl = self.dice_calc(pred, label)
                self.metrics[i+1] += list(dice_lvl.flatten().cpu().numpy())

            # if self.global_rank==0 and batch_idx <= 1:
            #     img_grid = torchvision.utils.make_grid(images[:40, ...])
            #     pred_grid = torchvision.utils.make_grid(torch.cat([preds[:40, ...],label[:40,...],torch.zeros_like(label[:40,...]) ], 1))
            #     plot_2d_or_3d_image(data=torch.concat([img_grid, pred_grid], axis=-1).unsqueeze(0), max_channels=3, step=self.current_epoch, writer= self.logger.experiment, tag=f"pred_{stage}_{batch_idx}")

        recon_loss = self.calc_losses(images, preds, label, stage)
        return recon_loss

    def on_validation_epoch_start(self):
        self.metrics = {i+1: [] for i in range(self.hparams.compute_capacity)}
    
    def on_validation_epoch_end(self):
        dice = 0
        for k,v in self.metrics.items():
            self.log(f"dice_level_{k}", np.mean(v), batch_size=1 , sync_dist=True)
            dice += np.mean(v)
        return {'dice': dice / len(self.metrics)}
    
    def get_metrics(self):
        metrics_dict = {}
        metrics_dict.update({f'dice_{k}': np.mean(v) for k,v in self.metrics.items()})
        metrics_dict.update({f'std_{k}': np.std(v) for k,v in self.metrics.items()})
        metrics_dict.update({'metr': self.metrics})

        return self.model.state_dict(), metrics_dict
  