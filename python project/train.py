import pickle as pickle
import os
import random
import argparse
import sys

import numpy as np
import pandas as pd
import json

from transformers import EarlyStoppingCallback

import torch
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments

from transformers import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, XLMRobertaForSequenceClassification, ElectraForSequenceClassification

import logging

import wandb

from sklearn.model_selection import train_test_split

def __get_logger():
    """로거 인스턴스 반환
    """

    __logger = logging.getLogger('logger')

    # # 로그 포멧 정의
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger

# seed설정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ParameterError(Exception):
    def __init__(self):
        super().__init__('Enter essential parameters')

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train():

  logger = __get_logger()

  logger.info("*************SETTING************")
  logger.info(f"model_name : {CFG.model_name}")
  logger.info(f"seed : {CFG.seed}")
  logger.info(f"scheduler : {CFG.scheduler}")
  logger.info(f"epochs : {CFG.epochs}")
  logger.info(f"ent_token : {CFG.ent_token}")
  logger.info(f"max_token_len : {CFG.max_token_len}")
  logger.info("********************************\n")

  #tokenizer설정
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  hyperparameter_defaults = dict(
    # learning_rate = CFG.learning_rate,
    epochs = CFG.epochs,
    # model_name = CFG.model_name,
    # batch_size = CFG.batch_size,
    # warmup_steps = CFG.warmup_steps,
    # weight_decay = CFG.weight_decay,
    )

  # # wandb설정
  # # wandb.login()
  wandb.init(config=hyperparameter_defaults, project="last")
  config = wandb.config



  if torch.cuda.is_available():
    logger.info("*************************************")
    device = torch.device("cuda")
    logger.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    logger.info(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
    logger.info("*************************************\n")
  else:
    logger.info("*************************************")
    logger.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    logger.info("*************************************\n")

  # model설정
  if CFG.model_name == 'kykim/bert-kor-base':
    model = BertForSequenceClassification.from_pretrained(CFG.model_name, num_labels = 42)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
  elif CFG.model_name == "monologg/koelectra-base-v3-discriminator":
    model = ElectraForSequenceClassification.from_pretrained(CFG.model_name, num_labels = 42)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model.to(device)
  elif CFG.model_name == "xlm-roberta-large":
    model =  XLMRobertaForSequenceClassification.from_pretrained(CFG.model_name, num_labels = 42)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model.to(device)
  

  if CFG.ent_token and CFG.model_name != "xlm-roberta-large":
    logger.info("bert/electra add token...")
    added_token_num = 0
    
    added_token_num += tokenizer.add_special_tokens({"additional_special_tokens":["[ENT]", "[/ENT]"]}) # 새로운 스페셜 토큰 추가 방법
    
    logger.info("*************************************")
    logger.info(f"add token number : {added_token_num}")
    logger.info(model.get_input_embeddings())
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num) # 모델의 embedding layer 층 개수 늘려 주워야 함
    logger.info(model.get_input_embeddings())
    logger.info("*************************************\n")
  # elif CFG.ent_token and CFG.model_name == "xlm-roberta-large":  # roberta에서 entity구분 토큰을 스페셜로 추가 할지 안할지 결정
  #   logger.info("roberta add token...")
  #   added_token_num = 0
    
  #   added_token_num += tokenizer.add_special_tokens({"additional_special_tokens":["#", "@", '₩', '^']}) # 새로운 스페셜 토큰 추가 방법
    
  #   logger.info("*************************************")
  #   logger.info(f"add token number : {added_token_num}")
  #   logger.info(model.get_input_embeddings())
  #   model.resize_token_embeddings(tokenizer.vocab_size + added_token_num) # 모델의 embedding layer 층 개수 늘려 주워야 함
  #   logger.info(model.get_input_embeddings())
  #   logger.info("*************************************\n")


  # load dataset
  train_dataset = load_data("/opt/ml/input/data/train/train_renew.tsv")
  # dev_dataset = load_data("/opt/ml/input/data/train/EDA/aug_train_EDA(test).tsv")
  # train_dataset, dev_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)
  # train_dataset = load_data("/opt/ml/input/data/train/k-fold2/train_split_0.tsv")
  # dev_dataset = load_data("/opt/ml/input/data/train/k-fold2/test_split_0.tsv")
  train_label = train_dataset['label'].values
  # dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer,CFG.ent_token, CFG.max_token_len, CFG.model_name)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer,CFG.ent_token, CFG.max_token_len, CFG.model_name)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label, CFG.ent_token, tokenizer, CFG.model_name)
  # RE_dev_dataset = RE_Dataset(tokenized_dev,dev_label, CFG.ent_token, tokenizer, CFG.model_name)

  print("\n*********tokenized example*********")
  print("tokenized sentence")
  print(tokenized_train[0].tokens)
  print()
  check_dataset = RE_train_dataset[0]
  print("input_ids")
  print(check_dataset["input_ids"])
  if CFG.model_name != 'xlm-roberta-large':
    print()
    print("token_type_ids")
    print(check_dataset["token_type_ids"])
  print()
  print("attention_mask")
  print(check_dataset["attention_mask"])
  if CFG.ent_token and CFG.model_name != "xlm-roberta-large":
    print()
    print("ent_ids")
    print(check_dataset["ent_ids"])
  print("***********************************\n")
  

  # early_stopping = EarlyStoppingCallback(early_stopping_patience = 5, early_stopping_threshold = 0.001)

  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
  # https://huggingface.co/transformers/_modules/transformers/trainer_utils.html - optimizer
  training_args = TrainingArguments(
    # report_to = 'wandb',              # wandb를 위한설정
    # run_name = CFG.save_path,         # wandb저장소 이름
    output_dir= CFG.save_path + 'results',          # output directory
    save_total_limit=2,              # number of total save model.
    save_steps=CFG.save_step,                 # model saving step.
    num_train_epochs=config.epochs,              # total number of training epochs s
    learning_rate=CFG.learning_rate,               # learning_rate
    per_device_train_batch_size=CFG.batch_size,  # batch size per device during training
    # per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=CFG.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=CFG.weight_decay,               # strength of weight decay
    logging_dir= CFG.save_path  + 'logs',            # directory for storing logs
    logging_steps=100,              # log saving step.(터미널에도 로그를 찍어주는 step)
    lr_scheduler_type = CFG.scheduler, # scheduler
    save_strategy='epoch',
    # label_smoothing_factor = 0.1,
    # evaluation_strategy='epoch', # evaluation strategy to adopt during training
    # fp16=True,  # 추가 부분
    # dataloader_num_workers=4, # 추가 부분
    # load_best_model_at_end= True, # 추가 부분
    # metric_for_best_model='accuracy'  # 추가 부분
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    #eval_steps = 500,            # evaluation step.
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    # eval_dataset=RE_dev_dataset,             # evaluation dataset
    # compute_metrics=compute_metrics,         # define metrics function
    # callbacks=[early_stopping]
  )

  # train model
  trainer.train()

def main():
  train()

class CFG:
  model_type = "bert"
  model_name = "bert-base-multilingual-cased"
  seed = 42
  scheduler = "linear"
  save_path = "./"
  epochs = 20 
  save_step = 500
  ent_token = False
  max_token_len = 100
  batch_size = 16
  warmup_steps = 500
  weight_decay = 0.01
  loss_type = "ce"
  learning_rate = 5e-5
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('-c', '--config', default=None, type=str,help='config file path')
  parser.add_argument('-s', '--save', default=None, type=str,help='save path')

  args = parser.parse_args()

  sys.path.append('/opt/ml/python project')
  from dataloader.load_data import *
  from models.models import *
  from util.loss import *

  if args.config == None or args.save == None:
    raise ParameterError

  # get hyperparam
  with open(args.config) as json_file:
    json_data = json.load(json_file)

  CFG.model_name = json_data['model_name']
  CFG.seed = json_data['seed']
  CFG.scheduler = json_data["scheduler"]
  CFG.save_path = args.save
  CFG.epochs = json_data["epochs"]
  CFG.save_step = json_data["save_step"]
  CFG.ent_token = json_data["ent_token"]
  CFG.max_token_len = json_data["max_token_len"]
  CFG.batch_size = json_data["batch_size"]
  CFG.wramup_steps = json_data["warmup_steps"]
  CFG.weight_decay = json_data["weight_decay"]
  CFG.loss_type = json_data["loss_type"]
  CFG.learning_rate = json_data["learning_rate"]
  seed_everything(CFG.seed)


  main()
