# For Bart, PEAGSUS one doc to one summary

OUTPUT_DIR=/scratch/xxu/multi-multi/data_single_trunk_1000/
mkdir ${OUTPUT_DIR}

python data_preprocess.py \
	-output_dir ${OUTPUT_DIR} \
	-mode one_to_one \
	-max_len_doc 1000 
	
#-tokenizer_model_path 'allenai/led-base-16384' \
