import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import  Trainer, TrainingArguments
from torch.utils.data import DataLoader
# from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import json
from tqdm import tqdm

def inference(model, tokenized_sent, device):
  """단일 모델 추론을 위한 함수

  Args:
    model : 평가 모델
    tokenized_sent : 토큰화 된 데이터 셋
    device  : 현재 device

  Returns:
    logits : 평가 lodits을 numpy형태로 저장(메모리 절약)
    predictions : 예측 값
  """
  logits = []
  predictions = []

  dataloader = DataLoader(tokenized_sent, batch_size=1, shuffle=False)
  model.eval()

  for i, data in enumerate(dataloader):

    with torch.no_grad():
      if 'token_type_ids' in data.keys():
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
      else:
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device)
          )

      _logits = outputs[0].detach().cpu().numpy()      
      _predictions = np.argmax(_logits, axis=-1)

      logits.append(_logits)
      predictions.extend(_predictions.ravel())

  return np.concatenate(logits), np.array(predictions)

def inference_multitask(model1,model2, tokenized_sent, device):
  """다중 모델 추론을 위한 함수

  Args:
    model1 : 평가 모델1 - 관계 있음, 관계 없음 2중 분류
    model2 : 평가 모델2 - 관계 있는 클래스에 한해서 클래스 분류
    tokenized_sent : 토큰화 된 데이터 셋
    device  : 현재 device

  Returns:
    logits : 평가 lodits을 numpy형태로 저장(메모리 절약)
    predictions : 예측 값
  """
  dataloader = DataLoader(tokenized_sent, batch_size=1, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      if 'token_type_ids' in data.keys():
        outputs = model1(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
      else:
        outputs = model1(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)
      
    # multi task를 위한 코드
    if result[0] == 0:
      output_pred.append(result)
    else:
      with torch.no_grad(): # 관계 있을 경우 두번째 모델로써 판단 
        if 'token_type_ids' in data.keys():
          outputs = model2(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device)
            )
        else:
          outputs = model2(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device)
            )
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()
      result = np.argmax(logits, axis=-1) + 1
      output_pred.append(result)
  
  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer, ent_token, max_token_len, model_name):
  """데이터셋을 파일에서 불러오는 함수

  Args:
    dataset_dir(str) : 평가하고자 하는 파일의 경로
    tokenizer : 모델의 토크나이저
    ent_token(bool) : entity token의 추가 여부
    max_token_len(int) : 토큰화 한 문장의 최대 길이
    model_name : 현재 모델의 이름

  Returns:
     tokenized_test(transformers.tokenization_utils_base.BatchEncoding) : 테스트 문장의 토큰화 된 결과
     test_label : 테스트 문장의 라벨
  """
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer,ent_token, max_token_len, model_name)
  return tokenized_test, test_label

def main(args):
  """주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드
  """
  if torch.cuda.is_available():
    print("\n*************************************")
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    print("*************************************\n")
  else:
    print("\n*************************************")
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    print("*************************************\n")

  # model설정
  model =  AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels = 42)
  # model2 =  XLMRobertaForSequenceClassification.from_pretrained("./classification2/results/checkpoint-1105", num_labels = 41)
  model.to(device)
  # model2.to(device)

  # tokinizer설정
  tokenizer = AutoTokenizer.from_pretrained(CFGInference.model_name)

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir,tokenizer, CFGInference.ent_token,CFGInference.max_token_len, CFGInference.model_name)
  test_dataset = RE_Dataset(test_dataset ,test_label, CFGInference.ent_token, tokenizer, CFGInference.model_name)

  # predict answer
  logits, predictions = inference(model, test_dataset, device)
    
  output = pd.DataFrame(predictions, columns=['pred'])
  output.to_csv('./prediction/submission.csv', index=False)
  np.save('./prediction/logits.npy', logits)

class CFGInference:
  model_name = "bert-base-multilingual-cased"
  ent_token = False
  max_token_len = 100

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/checkpoint-500")
  parser.add_argument('-c', '--config', default=None, type=str,help='config file path')
  args = parser.parse_args()

  sys.path.append('/opt/ml/backup')
  from dataloader.load_data import *

  # get hyperparam
  with open(args.config) as json_file:
    json_data = json.load(json_file)

  CFGInference.model_name = json_data['model_name']
  CFGInference.ent_token = json_data['ent_token']
  CFGInference.max_token_len = json_data['max_token_len']

  print("\n*************SETTING************")
  print("model_name : ", CFGInference.model_name)
  print("ent_token : ", CFGInference.ent_token)
  print("max_token_len : ", CFGInference.max_token_len)
  print("********************************\n")


  print(args)
  main(args)
  
