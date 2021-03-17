#!/bin/bash

DATA_PATH=/scratch/xxu/multi-multi/data/multi
MODEL_PATH=/scratch/xxu/multi-multi/models

python train.py \
	-mode validate \
	-data_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-pad_id 1 \
	-max_pos 800 \
	-batch_size 140 \
	-visible_gpus 0 \
	-log_file ../logs/val_abs_bert_cnndm \
	-result_path ../logs/abs_bert_cnndm
