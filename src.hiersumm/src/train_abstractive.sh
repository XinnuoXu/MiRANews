DATA_PATH=/scratch/xxu/multi-multi/data/multi
VOCAB_PATH=./multi.model
MODEL_PATH=/scratch/xxu/multi-multi/model.hiersumm
LOG_PATH=../log/

python train_abstractive.py \
	-data_path ${DATA_PATH} \
	-mode train \
	-batch_size 10000 \
	-seed 666 \
	-train_steps 50000 \
	-save_checkpoint_steps 2000 \
	-report_every 50 \
	-trunc_tgt_ntoken 400 \
	-trunc_src_nblock 24 \
	-visible_gpus 0,1,2 \
	-gpu_ranks 0,1,2 \
	-world_size 3 \
	-accum_count 4 \
	-dec_dropout 0.1 \
	-enc_dropout 0.1 \
	-label_smoothing 0.1 \
	-vocab_path ${VOCAB_PATH} \
	-model_path ${MODEL_PATH} \
	-accum_count 4 \
	-log_file ${LOG_PATH} \
	-inter_layers 6,7 \
	-inter_heads 8 \
	-hier
