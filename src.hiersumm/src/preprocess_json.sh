#!/bin/bash

RAW_PATH=/scratch/xxu/multi-multi/data_multi_hier/multi_
JSON_PATH=/scratch/xxu/multi-multi/json/multi

rm ${JSON_PATH}/*

python preprocess.py \
	-mode pretrain_to_json \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 30 \
	-log_file ../logs/multi_json.log \
