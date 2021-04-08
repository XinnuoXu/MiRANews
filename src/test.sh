#!/bin/bash
/bin/hostname -s
python3 -m torch.distributed.launch \
	--nproc_per_node=$NPROC_PER_NODE \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--node_rank=$SLURM_PROCID \
	--master_addr="$PARENT" --master_port="$MPORT" \
	run_summarization.py \
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
        --test_path data/multi \
	--num_beams 6 \
