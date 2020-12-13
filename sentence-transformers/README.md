## Sentence SegaBERT

Before training/testing, make sure our code is using segatron/transformers instead of  huggingface's transformers. In order to do that, adding `sys.path.insert(0,'path to segatron/transformers')` to `segatron_aaai/sentence_transformers/examples/evaluation_stsall.py`, `segatron_aaai/sentence_transformers/examples/training_nli.py`, and `segatron_aaai/sentence_transformers/sentence_transformers/models/Transformers.py`.

## Training Sentence SegaBERT

```
cd examples/training_transformers
```

```bash
python ./sentence_transformers/examples/training_transformers/training_nli.py --sega --model_path=[your pretrained model file path] --gpus=0 
```

## Testing with STS tasks

```
python ./sentence_transformers/examples/training_transformers/evaluation_stsall.py --sega --model_path=[the path of model from last command ] --gpus=0 
```

