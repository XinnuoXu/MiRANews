#!/bin/bash

# Multi-multi single
#INPUT_DIR=../data/data.multi/multi
#OUTPUT_DIR=./tmp/multi-summarization
#MAX_SORCE_LEN=1024

# Multi-multi cluster
#INPUT_DIR=../data/data.supp/multi
#OUTPUT_DIR=./tmp/multi-supp
#MAX_SORCE_LEN=1024

# Multi-multi rank
INPUT_DIR=../data/data.rank/multi
OUTPUT_DIR=./tmp/multi-rank
MAX_SORCE_LEN=1024

/bin/hostname -s
python3 -m torch.distributed.launch \
	--nproc_per_node=$NPROC_PER_NODE \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--node_rank=$SLURM_PROCID \
	--master_addr="$PARENT" --master_port="$MPORT" \
	run_summarization.py \
	--model_name_or_path google/bigbird-roberta-base \
        --do_train \
        --train_path ${INPUT_DIR} \
        --text_column text \
        --summary_column summary \
        --output_dir ${OUTPUT_DIR} \
	--max_source_length=${MAX_SORCE_LEN} \
	--max_target_length=128 \
	--num_train_epochs=12 \
	--group_by_length=true \
	--learning_rate=1e-04 \
	--weight_decay=0.01 \
	--max_grad_norm=0.1 \
	--label_smoothing_factor=0.1 \
	--lr_scheduler_type=polynomial \
	--attention_dropout=0.1 \
	--dropout=0.1 \
	--warmup_steps=500 \
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
	#--eval_steps=3000 \
	#--evaluation_strategy=steps \
	#--metric_for_best_model=loss \
	#--load_best_model_at_end=true \
	#--greater_is_better=false \
