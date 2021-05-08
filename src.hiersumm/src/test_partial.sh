DATA_PATH=/scratch/xxu/multi-multi/data/multi
VOCAB_PATH=./multi.model
MODEL_PATH=/scratch/xxu/multi-multi/model.hiersumm.partial/
LOG_PATH=../log/

python train_abstractive.py \
	-data_path ${DATA_PATH} \
	-mode validate \
	-test_all \
	-batch_size 30000 \
	-valid_batch_size 7500 \
	-seed 666 \
	-trunc_tgt_ntoken 400 \
	-trunc_src_nblock 40 \
	-visible_gpus 1 \
	-gpu_ranks 1 \
	-vocab_path ${VOCAB_PATH} \
	-model_path ${MODEL_PATH} \
	-log_file ${LOG_PATH} \
	-inter_layers 6,7 \
	-inter_heads 8 \
	-hier \
	-attn_main \
	-report_rouge \
	-max_wiki 100000 \
	-dataset test \
	-alpha 0.4 \
	-enc_dropout 0.1 \
	-max_length 400
	
