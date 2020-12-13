## Segatron Pretraining

First , download and process wikipedia/bookcorpus data with [wikiextractor](https://github.com/attardi/wikiextractor) and [BookCorpus](https://github.com/butsugiri/homemade_bookcorpus). For bookcorpus, we further process it into the output format of wikiextractor and the script is in `./bookcorpus/preprocess.ipynb`.

Then, run the following command to generate sentence splited data for pretraining.

```
./scripts/presplit_sentences_json.py
```

It should be noticed that the processed output file path should as same as the path in `data_utils/corpora.py`

Then, run the following command for segabert training. 

```
./scripts/pretrain_segabert_distributed.sh
```

The default parameters in this bash file are for the large model.

