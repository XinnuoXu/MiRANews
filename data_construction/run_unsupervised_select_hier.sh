# For Bart, PEAGSUS one doc to one summary

OUTPUT_DIR=/scratch/xxu/multi-multi/data_unsup_select_hier/
JSON_PATH=/scratch/xxu/multi-multi/json_unsup_select_hier/

mkdir ${OUTPUT_DIR}
mkdir ${JSON_PATH}
rm -rf ${OUTPUT_DIR}/*.jsonl
rm -rf ${JSON_PATH}/*.json

python data_preprocess.py \
	-potential_model_path 'allenai/led-base-16384' \
	-output_dir ${OUTPUT_DIR} \
	-save_path ${JSON_PATH} \
	-mode hier_unsupervised_select \
	-max_len_paragraph 150 \
	-max_num_paragraph 24 \
