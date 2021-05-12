# For LED BigBird one doc to one summary

OUTPUT_DIR=/scratch/xxu/multi-multi/data_single_trunk_5000/
JSON_PATH=/scratch/xxu/multi-multi/json_single_trunk_5000/

mkdir ${OUTPUT_DIR}
mkdir ${JSON_PATH}
rm -rf ${OUTPUT_DIR}/*.jsonl
rm -rf ${JSON_PATH}/*.json

python data_preprocess.py \
	-potential_model_path 'allenai/led-base-16384' \
	-output_dir ${OUTPUT_DIR} \
	-save_path ${JSON_PATH} \
	-mode one_to_one \
	-max_len_doc 5000 
	
#-tokenizer_model_path 'allenai/led-base-16384' \
