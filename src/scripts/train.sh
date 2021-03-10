#!/bin/bash

DATA_PATH=/scratch/xxu/multi-multi/data/multi
MODEL_PATH=/scratch/xxu/multi-multi/models

python train.py  \
	-mode train \
	-data_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-lr 0.001 \
	-pad_id 1 \
	-max_pos 800 \
	-adam_eps 1e-08 \
	-weight_decay 0.01 \
	-accum_count 5 \
	-batch_size 3 \
	-warmup_steps 20 \
	-train_epochs 2 \
	-val_check_interval 0.1 \
	-log_every_n_steps 5 \
	#-train_from ${MODEL_PATH}/model_step_16000.pt \
	#-warmup_steps 2000 \
