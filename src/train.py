#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import os
import argparse
from others.logging import init_logger
import pytorch_lightning as pl
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
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument("-num_dataload_workers", default=0, type=int)

    parser.add_argument("-finetune_bart", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-pad_id", default=0, type=int)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-max_pos", default=512, type=int)

    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-seed', default=777, type=int)
    parser.add_argument("-train_epochs", default=1000, type=int)
    parser.add_argument("-batch_size", default=140, type=int)
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
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.mode == 'train'):
        # Data_loader
        train_loader = LightningDataObject(args)
        # Checkpoint
        if args.train_from != '':
            logger.info('Loading checkpoint from %s' % args.train_from)
            checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
            opt = vars(checkpoint['opt'])
        else:
            checkpoint = None
        # Init Object
        train_obj = LightningObject(args, device, checkpoint)
        trainer = pl.Trainer(gpus=args.gpu_ranks, 
                            accelerator=args.lightning_accelerator, 
                            max_epochs=args.train_epochs,
                            val_check_interval=args.val_check_interval,
                            accumulate_grad_batches=args.accum_count)
        trainer.fit(train_obj, train_loader)

    '''
    elif (args.mode == 'validate'):
        validation(args, device_id)
    elif (args.mode == 'test'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test(args, device_id, cp, step)
    '''
