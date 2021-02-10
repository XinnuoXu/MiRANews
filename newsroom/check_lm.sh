python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file check_lm_train.txt \
    --validation_file check_lm_test.txt \
    --do_train \
    --do_eval \
    --output_dir ./tmp/test-clm
