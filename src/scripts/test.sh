#!/bin/bash

DATA_PATH=/scratch/xxu/multi-multi/data/multi
MODEL_PATH=/scratch/xxu/multi-multi/models

python train.py \
	-mode test \
	-data_path ${DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_10000.pt \
	-max_pos 800 \
	-alpha 0.9 \
	-min_length 20 \
	-max_length 100 \
	-test_batch_size 100 \
	-visible_gpus 0 \
	-log_file ../logs/val_abs_bert_cnndm \
	-result_path ../logs/abs_bert_cnndm
