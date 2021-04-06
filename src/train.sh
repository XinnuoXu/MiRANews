#!/bin/bash
/bin/hostname -s
python3 -m torch.distributed.launch \
	--nproc_per_node=$NPROC_PER_NODE \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--node_rank=$SLURM_PROCID \
	--master_addr="$PARENT" --master_port="$MPORT" \
	run_summarization.py \
	--model_name_or_path facebook/bart-large \
        --do_train \
        --train_path data/multi \
        --text_column text \
        --summary_column summary \
        --output_dir ./tmp/xsum-summarization \
	--max_source_length=512 \
	--max_target_length=128 \
	--num_train_epochs=12 \
	--group_by_length=true \
	--learning_rate=3e-05 \
	--weight_decay=0.01 \
	--max_grad_norm=0.1 \
	--label_smoothing_factor=0.1 \
	--lr_scheduler_type=polynomial \
	--attention_dropout=0.1 \
	--dropout=0.1 \
	--warmup_steps=500 \
	--gradient_accumulation_steps=8 \
        --per_device_train_batch_size=4 \
        --per_device_eval_batch_size=4 \
        --overwrite_output_dir \
        --predict_with_generate \
        --do_eval \
        --validation_path data/multi \
	--evaluation_strategy=steps \
	--eval_steps=3000 \
	--metric_for_best_model=loss \
	--load_best_model_at_end=true \
	--greater_is_better=false \
	--save_steps=2000 \
	--logging_steps=50 \
