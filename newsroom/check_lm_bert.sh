export CUDA_VISIBLE_DEVICES=0

EPOCH=30
LR=1e-5

#VAL_FILE=check_lm_test_same.txt
VAL_FILE=check_lm_test_cross.txt

python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file check_lm_train.txt \
    --validation_file ${VAL_FILE} \
    --per_gpu_train_batch_size 8 \
    --num_train_epochs ${EPOCH} \
    --learning_rate ${LR} \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --output_dir ./tmp/test-clm
