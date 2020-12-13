#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

WORKDIR="~/project/transformer-xl-master/pytorch/wiki103/$2"
if [[ $3 == 'train' ]]; then
    echo 'Run training...'
    python -u train.py \
        --cuda \
        --data ~/dataset/LM_data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --tgt_len 150 \
        --mem_len 150 \
        --attn_type 0 \
        --eval_tgt_len 150 \
        --batch_size 64 \
        --fp16 \
        --dynamic-loss-scale \
        --multi_gpu \
        --sega \
        --gpu0_bsz 16 \
        --work_dir ${WORKDIR}
elif [[ $3 == 'eval' ]]; then
    echo 'Run evaluation...'
    python -u eval.py \
        --cuda \
        --data ~/dataset/LM_data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 150 \
        --mem_len 150 \
        --batch_size 10 \
        --split test \
        --sega \
        --work_dir ${WORKDIR}
else
    echo 'unknown argment 1'
fi
