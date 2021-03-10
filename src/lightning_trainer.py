import os
import torch
from torch import nn
import pytorch_lightning as pl
from models.model_builder import FineTuneModel
from models.Schedulers import NoamLR
from models.Loaddata import SummDataset, batch_collate
from torch.utils.data import DataLoader

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

    def validation_step(self, batch, batch_idx):
        src = batch[0]
        tgt = batch[1].contiguous()
        mask_src = batch[2]
        mask_tgt = batch[3]
        labels = tgt[:, 1:].clone()
        labels[tgt[:, 1:] == self.pad_id] = -100
        loss, logits = self.model(src, tgt[:, :-1], mask_src, mask_tgt[:, :-1], labels)
        self.log('val_loss', loss)
        return {'loss': loss}

    def validation_step_end(self, batch_parts):
        if self.args.lightning_accelerator in ['dp', 'ddp2']:
            losses = [batch_parts[i]['loss'] for i in range(len(batch_parts))]
            return sum(losses)/len(losses)
        else:
            return batch_parts['loss']

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


class LightningDataObject(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        train_dataset = SummDataset(self.args, 'train', shuffle=True)
        train_loader = DataLoader(train_dataset,
                            batch_size=self.args.batch_size,
                            shuffle=True,
                            collate_fn=batch_collate,
                            num_workers=self.args.num_dataload_workers)
        return train_loader

    def val_dataloader(self):
        dev_dataset = SummDataset(self.args, 'dev', shuffle=True)
        dev_loader = DataLoader(dev_dataset,
                            batch_size=self.args.batch_size,
                            shuffle=True,
                            collate_fn=batch_collate,
                            num_workers=self.args.num_dataload_workers)
        return dev_loader

    def test_dataloader(self):
        test_dataset = SummDataset(self.args, 'test', shuffle=False)
        test_loader = DataLoader(test_dataset,
                            batch_size=self.args.batch_size,
                            shuffle=False,
                            collate_fn=batch_collate,
                            num_workers=self.args.num_dataload_workers)
        return test_loader

