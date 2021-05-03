# For Bart, PEAGSUS one doc to one summary

OUTPUT_DIR=/scratch/xxu/multi-multi/data_single_trunk_5000/
mkdir ${OUTPUT_DIR}

python data_preprocess.py \
	-output_dir ${OUTPUT_DIR} \
	-tokenizer_model_path '' \
	-mode one_to_one \
	-max_len_doc 5000 
