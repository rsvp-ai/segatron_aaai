import sys
sys.path.insert(0,'/home/user/project/segatron/sentence-transformers')
sys.path.insert(0,'/home/user/project/segatron/transformers')
from torch.utils.data import DataLoader
import math, random, argparse, os, re, json
import numpy as np
import torch
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.gpus.split(','))>0:
        torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sega', action='store_true')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_path', type=str, default='/home/user/output/bert_large_para_sent_token')
parser.add_argument('--gpus', type=str, required=True,
                        help='available gpus for training(separated by commas)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
# model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
set_seed(args)
# Read the dataset
train_batch_size = args.batch_size
model_save_path = args.model_path


model = SentenceTransformer(model_save_path)

folder = '../datasets/temp-sts/STS-data'
#'STS2012-gold','STS2013-gold','STS2014-gold','STS2015-gold',
names = ['STS2012-gold','STS2013-gold','STS2014-gold','STS2015-gold','STS2016-gold','SICK-data']

for name in names:

        sts_reader = STSDataReader(os.path.join(folder,name))
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples('all.tsv'),
                                                                          batch_size=train_batch_size,
                                                                          name=name+'-test')
        test_evaluator(model, output_path=model_save_path)

