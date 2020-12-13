"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
import sys
sys.path.insert(0,'/home/user/project/segatron/sentence-transformers')
sys.path.insert(0,'/home/user/project/segatron/transformers')
from torch.utils.data import DataLoader
import math, random, argparse, os
import numpy as np
import torch
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
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
parser.add_argument('--main_similarity', type=int, default=None,
                        help='metric for eval')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
model_name = args.model_path
set_seed(args)
# Read the dataset
train_batch_size = args.batch_size
nli_reader = NLIDataReader('../datasets/AllNLI')
sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark')
train_num_labels = nli_reader.get_num_labels()
model_save_path = 'output/'+model_name.split("/")[-1]+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name,sega=args.sega)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Convert the dataset to a DataLoader ready for training
logging.info("Read AllNLI train dataset")
train_dataset = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)


logging.info("Read STSbenchmark dev dataset")
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples('sts-dev.csv'),
                                                                 batch_size=train_batch_size, name='sts-dev',
                                                                 main_similarity=args.main_similarity)

# Configure the training
num_epochs = 1

warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples('sts-test.csv'), batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)
