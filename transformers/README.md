## Finetuning SegaBERT 

After pre-training, convert the model file to HuggingFace format with 

```bash
PYTHONPATH=absolute_path_of(./transformers/) python ./transformers/transformers/convert_segatron_to_huggingface.py --segatron_path <your model file path> --bert_config_file <bert config file path> --huggingface_path <target path>
```

### QQP MNLI QNLI

```bash
./transformers/examples/eval_finetune/train_glue.sh 0,1,2,3,4,5,6,7 16000
```

### Grid search for MRPC RTE CoLA STS-B SST-2

```bash
./transformers/examples/eval_finetune/grid_search_glue.sh 
```

### SQUAD v1.1 or v2.0

```bash
./transformers/examples/eval_finetune/train_squad.sh 0,1,2,3,4,5,6,7 16000
```

```bash
./transformers/examples/eval_finetune/train_squad2.sh 0,1,2,3,4,5,6,7  16000
```

## RACE

```bash
./transformers/examples/eval_finetune/run_race.sh 0,1,2,3,4,5,6,7 16000 
```

## 