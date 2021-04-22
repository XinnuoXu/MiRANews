#!/bin/bash

RAW_PATH=/scratch/xxu/multi-multi/raw_data/multi_
JSON_PATH=/scratch/xxu/multi-multi/json/multi

python preprocess.py \
	-mode pretrain_to_json \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 30 \
	-log_file ../logs/multi_json.log \
