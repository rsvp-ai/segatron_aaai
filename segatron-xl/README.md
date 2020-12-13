
## Data Prepration

`bash getdata.sh`

##Train

segaTransformer-xl base

```bash
bash run_wt103_base_sega.sh 0,1,2,3 sega_base train
```

segaTransformer-xl large

```bash
bash run_wt103_large_sega.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 sega_large train
```

It should be noticed that the training steps is 500K by defaults. We stopped the training process after 350K steps.

##Test

For base models, the testing would be conducted automatically when training finished.

For large models, as we stopped the training according to the evaluation results, run the following command for testing:

```
bash run_wt103_large_sega.sh 0 sega_large eval
```

