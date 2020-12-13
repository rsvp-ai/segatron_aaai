#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="/home/user/project/transformers"

MODEL_PATH="/home/user/output/segabert-large"
NUM_NODES=$((`echo $1 | tr -cd , | wc -c`+1))
python -m torch.distributed.launch --nproc_per_node=${NUM_NODES} --master_port=$2 ../run_squad.py \
--version_2_with_negative \
--model_type bert \
--model_name_or_path ${MODEL_PATH} \
--do_train \
--eval_all_checkpoints \
--do_eval \
--do_lower_case \
--train_file ./squad_data-v2.0/train-v2.0.json \
--predict_file ./squad_data-v2.0/dev-v2.0.json \
--learning_rate 3e-5 \
--num_train_epochs 4 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir ${MODEL_PATH}/squad2 \
--per_gpu_eval_batch_size=32 \
--per_gpu_train_batch_size=8 \
--save_steps=1000

