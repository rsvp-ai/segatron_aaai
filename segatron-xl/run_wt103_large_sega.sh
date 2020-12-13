#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

WORKDIR="~/project/transformer-xl-master/pytorch/wiki103/$2"

if [[ $3 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ~/dataset/LM_data/wikitext-103/ \
        --dataset wt103 \
        --restart \
        --restart_dir ${WORKDIR}-wt103 \
        --adaptive \
        --div_val 4 \
        --n_layer 18 \
        --d_model 1024 \
        --n_head 16 \
        --d_head 64 \
        --d_inner 4096 \
        --dropout 0.2 \
        --dropatt 0.2 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 16000 \
        --max_step 500000 \
        --tgt_len 384 \
        --mem_len 384 \
        --eval_tgt_len 384 \
        --batch_size 128 \
        --fp16 \
        --dynamic-loss-scale \
        --multi_gpu \
        --sega \
        --gpu0_bsz 8 \
        --work_dir ${WORKDIR}
elif [[ $3 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ~/dataset/LM_data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 384 \
        --mem_len 384 \
        --sega \
        --batch_size 1 \
        --split test \
        --work_dir ${WORKDIR}
else
    echo 'unknown argment 1'
fi
