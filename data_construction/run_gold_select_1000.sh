# For Bart, PEAGSUS one doc to one summary

INPUT_DIR=/scratch/xxu/multi-multi/supervised_content_labels/
OUTPUT_DIR=/scratch/xxu/multi-multi/data_gold_select_1000/
JSON_PATH=/scratch/xxu/multi-multi/json_gold_select_1000/

mkdir ${OUTPUT_DIR}
mkdir ${JSON_PATH}
rm -rf ${OUTPUT_DIR}/*.jsonl
rm -rf ${JSON_PATH}/*.json

python data_preprocess.py \
	-potential_model_path 'facebook/bart-large' \
	-tokenizer_model_path 'facebook/bart-large' \
	-root_dir ${INPUT_DIR} \
	-output_dir ${OUTPUT_DIR} \
	-save_path ${JSON_PATH} \
	-mode gold_select \
	-max_len_sup 500 \
	-max_len_doc 500
	
