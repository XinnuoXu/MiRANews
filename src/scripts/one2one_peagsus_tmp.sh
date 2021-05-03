#!/bin/bash

INPUT_DIR=/scratch/xxu/multi-multi/json_single_trunk_5000/multi
#INPUT_DIR=../data/json_single_trunk_1000/multi
OUTPUT_DIR=../saved_checkpoints/single_trunk_peagsus/
MAX_SORCE_LEN=1024

/bin/hostname -s
python3 run_summarization.py \
	--model_name_or_path 'google/pegasus-multi_news' \
        --do_train \
        --train_path ${INPUT_DIR} \
        --text_column text \
        --summary_column summary \
        --output_dir ${OUTPUT_DIR} \
	--max_source_length=${MAX_SORCE_LEN} \
	--max_target_length=256 \
	--num_train_epochs=15 \
	--group_by_length=true \
	--learning_rate=1e-04 \
	--weight_decay=0.01 \
	--max_grad_norm=0.1 \
	--label_smoothing_factor=0.1 \
	--lr_scheduler_type=polynomial \
	--attention_dropout=0.1 \
	--dropout=0.1 \
	--warmup_steps=1000 \
	--gradient_accumulation_steps=32 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --overwrite_output_dir \
        --predict_with_generate \
        --do_eval \
        --validation_path ${INPUT_DIR} \
	--evaluation_strategy=no \
	--save_steps=3000 \
	--logging_steps=50 \
	--local_files_only false \
