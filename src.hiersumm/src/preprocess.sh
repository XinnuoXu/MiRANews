#!/bin/bash

JSON_PATH=/scratch/xxu/multi-multi/json/
DATA_PATH=/scratch/xxu/multi-multi/data/

python preprocess.py \
	-mode pretrain_to_data \
	-raw_path ${JSON_PATH} \
	-save_path ${DATA_PATH} \
	-max_src_ntokens_per_sent 350 \
      	-lower \
	-n_cpus 30 \
	-log_file ../logs/preprocess.log