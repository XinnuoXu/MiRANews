#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from lightning_trainer import LightningObject, LightningDataObject

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-test_data", default='test', type=str, choices=['test'])
    parser.add_argument("-data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-train_from", default='')
    parser.add_argument("-train_state", default='')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-num_dataload_workers", default=0, type=int)

    parser.add_argument("-finetune_bart", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-pad_id", default=0, type=int)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-max_pos", default=512, type=int)

    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument("-num_nodes", default=1, type=int)

    parser.add_argument('-seed', default=777, type=int)
    parser.add_argument("-train_epochs", default=1000, type=int)
    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-log_every_n_steps", default=50, type=int)
    parser.add_argument("-val_check_interval", default= 0.5, type=float)
    parser.add_argument("-lightning_accelerator", default='ddp', type=str, choices=['dp', 'ddp', 'ddp2'])

    # optimizer
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-adam_eps", default=1e-9, type=float)
    parser.add_argument("-weight_decay", default=0.0, type=float)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-attention_dropout", default=0.1, type=float)

    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    # testing parameters
    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)

    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    if args.visible_gpus == '-1':
        args.gpu_ranks = -1
    else:
        args.gpu_ranks = [int(gpu_id) for gpu_id in args.visible_gpus.split(',')]
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.mode == 'train'):
        seed_everything(args.seed)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # Data_loader
        train_loader = LightningDataObject(args)
        # Init Object
        if args.train_from != '':
            train_obj = LightningObject.load_from_checkpoint(checkpoint_path=args.train_from, 
                                                                hparams_file=args.train_state,
                                                                args=args, 
                                                                device=device)
        else:
            train_obj = LightningObject(args, device)
        # Initialize checkpoint_callback
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                                filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
                                                save_top_k=3,
                                                mode='min')
        # Log
        trainer = pl.Trainer(gpus=args.gpu_ranks, 
                                num_nodes=args.num_nodes,
                                accelerator=args.lightning_accelerator, 
                                max_epochs=args.train_epochs,
                                val_check_interval=args.val_check_interval,
                                accumulate_grad_batches=args.accum_count,
                                callbacks=[lr_monitor, checkpoint_callback],
                                log_every_n_steps=args.log_every_n_steps)
        trainer.fit(train_obj, train_loader)

    elif (args.mode == 'test'):
        test_loader = LightningDataObject(args)
        model = LightningObject.load_from_checkpoint(checkpoint_path=args.test_from,
                                                    args=args,
                                                    device=device)
        trainer = pl.Trainer(gpus=args.gpu_ranks, num_nodes=args.num_nodes)
        trainer.test(model=model, datamodule=test_loader)
