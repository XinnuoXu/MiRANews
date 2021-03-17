#!/bin/bash

DATA_PATH=/scratch/xxu/multi-multi/data/multi
MODEL_PATH=/scratch/xxu/multi-multi/models

python train.py  \
	-mode train \
	-data_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-lr 1e-03 \
	-pad_id 1 \
	-max_pos 800 \
	-adam_eps 1e-08 \
	-weight_decay 0.01 \
	-accum_count 10 \
	-batch_size 3 \
	-warmup_steps 150 \
	-train_epochs 35 \
	-val_check_interval 1.0 \
	-log_every_n_steps 50 \
	-visible_gpus 0,1,2 \
	#-train_from ./lightning_logs/version_0/checkpoints/sample-mnist-epoch=00-val_loss=8.41.ckpt \
	#-train_state ./lightning_logs/version_0/hparams.yaml \
