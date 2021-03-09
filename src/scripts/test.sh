#!/bin/bash

DATA_PATH=/scratch/xxu/multi-multi/data/multi
MODEL_PATH=/scratch/xxu/multi-multi/models

python train.py \
	-mode test \
	-data_path ${DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_30000.pt \
	-pad_id 1 \
	-max_pos 800 \
	-alpha 0.9 \
	-beam_size 5 \
	-min_length 10 \
	-max_length 200 \
	-batch_size 1024 \
	-visible_gpus 0 \
	-log_file ../logs/val_abs_bert_cnndm \
	-result_path ../logs/abs_bert_cnndm
