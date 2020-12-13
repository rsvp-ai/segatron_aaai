#!/bin/bash

# Default config for 16 GPUs single node machine
GPUS_PER_NODE=16
MASTER_ADDR=127.0.0.1
MASTER_PORT=16010
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

PYTHONPATH="path of pretrain_segabert.py's folder"
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_segabert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 24 \
       --batch-size 16 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-para-position-embeddings 50 \
       --max-sent-position-embeddings 100 \
       --max-token-position-embeddings 512 \
       --train-iters 1000000 \
       --ds-type BERT_PST \
       --save /home/baihe/output/segabert_large \
       --tensorboard-dir /home/baihe/output/segabert_large \
       --resume-dataloader \
       --train-data wikipedia_pos bookcorpus_pos \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-base-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 900000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-layernorm \
       --fp32-embedding
