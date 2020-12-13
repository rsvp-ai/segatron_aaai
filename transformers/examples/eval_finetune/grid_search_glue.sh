#!/usr/bin/env bash
export PYTHONPATH="/home/user/project/transformers"
# CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE
TASK_LIST="MRPC RTE CoLA STS-B SST-2"
LR_LIST="2e-5 3e-5 5e-5"
BS_LIST="8 12 16"
EPOCH="6"

run_glue () {
    local DEVICES=$1
    local TASK_NAME=$2
    export CUDA_VISIBLE_DEVICES=$DEVICES
    NUM_NODES=$((`echo $DEVICES | tr -cd , | wc -c`+1))
    echo "$1 $2 --master_port=$3 ../run_glue.py"
    python -m torch.distributed.launch --nproc_per_node=${NUM_NODES} --master_port=$3 ../run_glue.py \
    --model_type segabert \
    --model_name_or_path ${FOLDER} \
    --do_train \
    --do_eval \
    --skip_evaled \
    --skip_trained \
    --eval_all_checkpoints \
    --do_lower_case \
    --data_dir ./glue_data/${TASK_NAME} \
    --task_name ${TASK_NAME} \
    --learning_rate $6 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs $4 \
    --max_seq_length 256 \
    --output_dir ${FOLDER}/${TASK_NAME}_$5_$6 \
    --per_gpu_eval_batch_size=256 \
    --per_gpu_train_batch_size=$5 \
    --per_gpu_test_batch_size=1024 \
    --save_steps=-1
}


cuda_index=0
port=15000
for TASK_NAME in $TASK_LIST
do
    FOLDER="/home/user/output/segabert_large"
    for lr in $LR_LIST
    do
        for bs in $BS_LIST
        do
            if [ $cuda_index -gt 7 ]
            then
                cuda=`python get_aviable_cuda.py --index $cuda_index`
            else
                cuda=$cuda_index
            fi
            run_glue "$cuda" "$TASK_NAME" "$port" "$EPOCH" "$bs" "$lr"&
            ((port++))
            ((cuda_index++))
        done
    done
done

#            if [ $cuda_index -gt 15 ]
#            then
#                sleep 60
#            fi