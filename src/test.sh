#!/bin/bash
python run_summarization.py \
	--output_dir ./tmp/xsum-test/ \
	--overwrite_output_dir \
        --text_column text \
        --summary_column summary \
	--per_device_eval_batch_size 1 \
        --do_predict \
	--model_name_or_path ./tmp/xsum-summarization/checkpoint-15000 \
	--max_source_length=512 \
	--max_target_length=128 \
	--val_max_target_length=60 \
        --test_path /scratch/xxu/multi-multi/json//multi \
	--num_beams 6 \
