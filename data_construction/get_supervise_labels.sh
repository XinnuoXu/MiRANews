# For Bart, PEAGSUS one doc to one summary

#OUTPUT_DIR=/scratch/xxu/multi-multi/supervised_content_labels/

python data_preprocess.py \
	-potential_model_path 'facebook/bart-large' \
	-gt_input $1 \
	-gt_output_src $2 \
	-gt_output_tgt $3 \
	-mode rouge_gt \
	-thred_num 3 \
	-max_len_doc 5000 
	
