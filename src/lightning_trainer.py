import os
import torch
from torch import nn
import pytorch_lightning as pl
from models.model_builder import FineTuneModel
from models.Schedulers import NoamLR

class LightningObject(pl.LightningModule):
    def __init__(self, args, device, checkpoint):
        super(LightningObject, self).__init__()
        self.args = args
        self.model = FineTuneModel(args, device, checkpoint)
        self.pad_id = args.pad_id

    def forward(self, batch):
        src = batch.src
        mask_src = batch.mask_src
        hypos = self.model.generate(src,
                    beam_size=self.args.beam_size,
                    max_length=self.args.max_length,
                    min_length=self.args.min_length,
                    early_stopping=True)
        return hypos

    def training_step(self, batch, batch_idx):
        src = batch[0]
        tgt = batch[1].contiguous()
        mask_src = batch[2]
        mask_tgt = batch[3]
        labels = tgt[:, 1:].clone()
        labels[tgt[:, 1:] == self.pad_id] = -100
        loss, logits = self.model(src, tgt[:, :-1], mask_src, mask_tgt[:, :-1], labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2),
                                    eps=self.args.adam_eps,
                                    weight_decay=self.args.weight_decay)
        scheduler = NoamLR(optimizer, self.args.warmup_steps)
        opt_obj = {'optimizer':optimizer,
                    'lr_scheduler':scheduler,
                    'interval': 'step',
                    'frequency': 1
                  }
        return opt_obj
