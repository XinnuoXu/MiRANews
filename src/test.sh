#!/bin/bash

# Multi-multi single
#DATA_PATH=data.multi/multi
#MODEL_PATH=./tmp/multi-summarization
#OUTPUT_DIR=./tmp/multi-test/

# Multi-multi supp
DATA_PATH=data.supp/multi
MODEL_PATH=./tmp/multi-supp
OUTPUT_DIR=./tmp/supp-test/

# XSum
#MODEL_PATH=./tmp/xsum-summarization/
#OUTPUT_DIR=./tmp/xsum-test/
#DATA_PATH=./data/multi

python run_summarization.py \
	--output_dir ${OUTPUT_DIR} \
	--overwrite_output_dir \
        --text_column text \
        --summary_column summary \
	--per_device_eval_batch_size 4 \
        --do_predict \
	--model_name_or_path ${MODEL_PATH} \
	--max_source_length=512 \
	--max_target_length=128 \
	--val_max_target_length=60 \
        --test_path ${DATA_PATH} \
	--num_beams 6 \
	--predict_with_generate \
