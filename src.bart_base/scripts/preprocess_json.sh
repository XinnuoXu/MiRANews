#!/bin/bash

RAW_PATH=/scratch/xxu/multi-multi/data_single_trunk_1000/multi_
#RAW_PATH=/home/xx6/Factroid_Summarization/XSum/bbc-summary-no_structure/xsum_
JSON_PATH=/scratch/xxu/multi-multi/json/multi

python preprocess.py \
	-mode pretrain_to_json \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 30 \
	-log_file ../logs/cnndm.log \
