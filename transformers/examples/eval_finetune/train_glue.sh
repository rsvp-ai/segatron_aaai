#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="/home/user/project/transformers"
MODEL_PATH="/home/user/output/segabert-large"
NUM_NODES=$((`echo $1 | tr -cd , | wc -c`+1))
# CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE
TASK_LIST="QQP MNLI QNLI"
for TASK_NAME in $TASK_LIST
do
    python -m torch.distributed.launch --nproc_per_node=${NUM_NODES} --master_port=$2 ../run_glue.py \
    --model_type segabert \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ./glue_data/${TASK_NAME} \
    --task_name ${TASK_NAME} \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length 256 \
    --output_dir ${MODEL_PATH}/${TASK_NAME} \
    --overwrite_output_dir \
    --per_gpu_eval_batch_size=128 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_test_batch_size=512 \
    --save_steps=-1 &&
    echo "==========$TASK_NAME finished!=========="
done