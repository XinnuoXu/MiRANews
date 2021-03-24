import os
import torch
import math
from torch import nn
import pytorch_lightning as pl
from models.model_builder import FineTuneModel
from models.Schedulers import NoamLR
from models.Inference import Translator
from models.Loaddata import SummDataset, batch_collate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from transformers import get_polynomial_decay_schedule_with_warmup

class LightningObject(pl.LightningModule):
    def __init__(self, args, device):
        super(LightningObject, self).__init__()
        self.args = args
        self.model = FineTuneModel(args, device)
        self.translator = Translator(args, self.model)
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
        self.log('ppl', math.exp(min(loss, 100)))
        self.log('train_loss', loss)
        normalization = labels[:, 1:].ne(-100).sum().item()
        return loss.div(float(normalization))

    def validation_step(self, batch, batch_idx):
        src = batch[0]
        tgt = batch[1].contiguous()
        mask_src = batch[2]
        mask_tgt = batch[3]
        labels = tgt[:, 1:].clone()
        labels[tgt[:, 1:] == self.pad_id] = -100
        loss, logits = self.model(src, tgt[:, :-1], mask_src, mask_tgt[:, :-1], labels)
        loss = loss * 100
        self.log('val_loss', loss)
        return {'loss': loss}

    def validation_step_end(self, batch_parts):
        if self.args.lightning_accelerator in ['dp', 'ddp2']:
            losses = [batch_parts[i]['loss'] for i in range(len(batch_parts))]
            return sum(losses)/len(losses)
        else:
            return batch_parts['loss']

    def test_step(self, batch, batch_idx):
        self.translator.translate(batch)
        return 0.0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2),
                                    eps=self.args.adam_eps,
                                    weight_decay=self.args.weight_decay)

        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=self.args.warmup_steps,
                                                            num_training_steps=self.args.train_steps)
        #scheduler = {'scheduler': NoamLR(optimizer, self.args.warmup_steps),
        scheduler = {'scheduler': scheduler,
                     'monitor': 'metric_to_track',
                     'interval': 'step',
                     'frequency': 1,
                     'strict': True,
                    }
        return [optimizer], [scheduler]


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

