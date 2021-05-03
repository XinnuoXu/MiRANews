#!/bin/bash
/bin/hostname -s
python3 -m torch.distributed.launch \
	--nproc_per_node=$NPROC_PER_NODE \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--node_rank=$SLURM_PROCID \
	--master_addr="$PARENT" --master_port="$MPORT" \
	run_summarization.py \
	--output_dir ./tmp/xsum-valid/ \
	--overwrite_output_dir \
	--model_name_or_path ./tmp/xsum-summarization/checkpoint-3000 \
        --text_column text \
        --summary_column summary \
	--max_source_length=512 \
	--max_target_length=128 \
        --per_device_eval_batch_size=64 \
        --do_eval \
        --validation_path data/multi \
