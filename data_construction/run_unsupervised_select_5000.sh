# For Bart, PEAGSUS one doc to one summary

OUTPUT_DIR=/scratch/xxu/multi-multi/data_unsup_select_5000/
JSON_PATH=/scratch/xxu/multi-multi/json_unsup_select_5000/

mkdir ${OUTPUT_DIR}
mkdir ${JSON_PATH}
rm -rf ${OUTPUT_DIR}/*.jsonl
rm -rf ${JSON_PATH}/*.json

python data_preprocess.py \
	-tokenizer_model_path 'facebook/bart-large' \
	-potential_model_path 'allenai/led-base-16384' \
	-output_dir ${OUTPUT_DIR} \
	-save_path ${JSON_PATH} \
	-mode unsupervised_select \
	-max_len_sup 2500 \
	-max_len_doc 2500 
	
