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
	-batch_size 140 \
	-report_every 50 \
	-warmup_steps 2000 \
	-train_steps 30000 \
	-visible_gpus 0 \
	-save_checkpoint_steps 5000 \
	-log_file ../logs/abs_bert_cnndm
