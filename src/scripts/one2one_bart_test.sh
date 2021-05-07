#!/bin/bash

# Multi-multi rank
DATA_PATH=../data/json_single_trunk_1000/multi
MODEL_PATH=../saved_checkpoints/single_trunk_bart/checkpoint-12000/
OUTPUT_DIR=../saved_results/single_trunk_bart/
MAX_SORCE_LEN=1024

#mkdir ${OUTPUT_DIR}

python run_summarization.py \
	--output_dir ${OUTPUT_DIR} \
	--overwrite_output_dir \
        --text_column text \
        --summary_column summary \
	--per_device_eval_batch_size 4 \
        --do_predict \
	--model_name_or_path ${MODEL_PATH} \
	--max_source_length=${MAX_SORCE_LEN} \
	--max_target_length=256 \
	--val_max_target_length=256 \
        --test_path ${DATA_PATH} \
	--num_beams 6 \
	--predict_with_generate \
