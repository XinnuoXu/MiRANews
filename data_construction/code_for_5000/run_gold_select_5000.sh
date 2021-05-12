# For Bart, PEAGSUS one doc to one summary

INPUT_DIR=/scratch/xxu/multi-multi/supervised_content_labels/
OUTPUT_DIR=/scratch/xxu/multi-multi/data_gold_select_5000/
JSON_PATH=/scratch/xxu/multi-multi/json_gold_select_5000/

mkdir ${OUTPUT_DIR}
mkdir ${JSON_PATH}
rm -rf ${OUTPUT_DIR}/*.jsonl
rm -rf ${JSON_PATH}/*.json

python data_preprocess.py \
	-potential_model_path 'allenai/led-base-16384' \
	-root_dir ${INPUT_DIR} \
	-output_dir ${OUTPUT_DIR} \
	-save_path ${JSON_PATH} \
	-mode gold_select \
	-max_len_sup 1000 \
	-max_len_doc 5000 \
	
