# For Bart, PEAGSUS one doc to one summary

OUTPUT_DIR=/scratch/xxu/multi-multi/data_single_trunk_1000/
JSON_PATH=/scratch/xxu/multi-multi/json_single_trunk_1000/

mkdir ${OUTPUT_DIR}
mkdir ${JSON_PATH}
rm -rf ${OUTPUT_DIR}/*.jsonl
rm -rf ${JSON_PATH}/*.json

python data_preprocess.py \
	-tokenizer_model_path 'facebook/bart-large' \
	-output_dir ${OUTPUT_DIR} \
	-save_path ${JSON_PATH} \
	-mode one_to_one \
	-max_len_doc 1000 
	
