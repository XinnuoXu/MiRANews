# For Hier

OUTPUT_DIR=/scratch/xxu/multi-multi/data_single_hier/

mkdir ${OUTPUT_DIR}
rm -rf ${OUTPUT_DIR}/*.jsonl

python data_preprocess.py \
	-tokenizer_model_path 'facebook/bart-large' \
	-output_dir ${OUTPUT_DIR} \
	-mode hier_one_to_one \
	-max_len_paragraph 200 \
