# For Hier

INPUT_DIR=/scratch/xxu/multi-multi/supervised_content_labels/
OUTPUT_DIR=/scratch/xxu/multi-multi/data_gold_select_hier/

mkdir ${OUTPUT_DIR}
rm -rf ${OUTPUT_DIR}/*.jsonl

python data_preprocess.py \
	-tokenizer_model_path 'facebook/bart-large' \
	-root_dir ${INPUT_DIR} \
	-output_dir ${OUTPUT_DIR} \
	-mode hier_gold_select \
	-max_len_paragraph 150 \
	-max_num_paragraph 24 \
	
