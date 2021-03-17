#!/bin/bash

DATA_PATH=/scratch/xxu/multi-multi/data/multi
MODEL_PATH=./lightning_logs/version_6/checkpoints/

python train.py \
	-mode test \
	-data_path ${DATA_PATH} \
	-test_from ${MODEL_PATH}/sample-mnist-epoch\=33-val_loss\=5.08.ckpt \
	-pad_id 1 \
	-max_pos 800 \
	-alpha 0.9 \
	-beam_size 5 \
	-min_length 10 \
	-max_length 200 \
	-batch_size 3 \
	-visible_gpus 0 \
	-result_path ../logs/abs_bert_cnndm
