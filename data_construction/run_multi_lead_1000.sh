# For Bart, PEAGSUS one doc to one summary

OUTPUT_DIR=/scratch/xxu/multi-multi/data_multi_lead_1000/
JSON_PATH=/scratch/xxu/multi-multi/json_multi_lead_1000/

mkdir ${OUTPUT_DIR}
mkdir ${JSON_PATH}
rm -rf ${OUTPUT_DIR}/*.jsonl
rm -rf ${JSON_PATH}/*.json

python data_preprocess.py \
	-potential_model_path 'facebook/bart-large' \
	-tokenizer_model_path 'facebook/bart-large' \
	-output_dir ${OUTPUT_DIR} \
	-save_path ${JSON_PATH} \
	-mode multi_to_one_lead \
	-max_len_sup 500 \
	-max_len_doc 500 
	
