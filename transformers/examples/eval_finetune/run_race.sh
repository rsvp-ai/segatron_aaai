#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="/home/user/project/transformers"

MODEL_PATH="/home/user/output/segabert-large"
NUM_NODES=$((`echo $1 | tr -cd , | wc -c`+1))

python -m torch.distributed.launch --nproc_per_node=${NUM_NODES} --master_port=$2 ../run_multiple_choice.py \
--model_type segabert \
--model_name_or_path ${MODEL_PATH} \
--data_dir=/home/user/project/transformers/examples/eval_finetune/RACE  \
--output_dir ${MODEL_PATH}/RACE \
--max_seq_length=512 \
--learning_rate=2e-5 \
--num_train_epochs=3 \
--fp16 \
--do_train \
--do_lower_case \
--task_name RACE \
--overwrite_output_dir \
--per_gpu_eval_batch_size=128 \
--per_gpu_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--save_steps=1000