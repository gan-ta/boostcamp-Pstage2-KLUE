import pickle as pickle
import os
import pandas as pd
import torch
from pororo import Pororo
from tqdm import tqdm

class RE_Dataset(torch.utils.data.Dataset):
  """학습에 쓰일 데이터셋 클래스
  """
  def __init__(self, tokenized_dataset, labels, ent_token, tokenizer, model_name):
    """
    Args:
      tokenized_dataset : 토큰화된 데이터 셋
      labels(list) : 클래스 라벨 정보
      ent_token(bool) : entity token에 대하여 아이디를 부여할지 확인
      tokenizer : model의 토크나이저
      model_name(str) : 모델의 이름
    """
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels
    self.ent_token = ent_token
    self.model_name = model_name
    
    # 추가 작업 - ENT vocab아이디를 가져옴 (ent_id를 사용하여 kobert layer의 input, output에 활용)
    self.start_ent_id = None
    self.end_ent_id = None
    if ent_token and self.model_name != "xlm-roberta-large":
      vocab = tokenizer.get_vocab()
      print("\n****************ENT ID***************")
      print(f"start ENT : {vocab['[ENT]']}")
      print(f"end ENT : {vocab['[/ENT]']}")
      print("*************************************\n")
      self.start_ent_id = vocab['[ENT]']
      self.end_ent_id = vocab['[/ENT]']
    elif ent_token and self.model_name == "xlm-roberta-large":
      vocab = tokenizer.get_vocab()
      print("\n****************ENT ID***************")
      print(f"token1 : {vocab['₩']}")
      print(f"token2 : {vocab['^']}")
      print(f"token3 : {vocab['#']}")
      print(f"token4 : {vocab['@']}")
      print("*************************************\n")

  def __getitem__(self, idx):
    """ ent_token에 따라 ent_id의 생성여부를 확인(이는 kobert에서만 사용)
    """
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])

    # input layer, output layer층을 튜닝하기 위한 작업
    if self.ent_token and self.model_name != "xlm-roberta-large":
      ent_ids = []
      flag = False
      count = 0
      for v in item['input_ids']:
        # ENT토큰은 Entity로 취급하지 않음
        if v == self.start_ent_id:
          flag = True
          count += 1
          ent_ids.append(0)
          continue
        elif v ==  self.end_ent_id:
          flag = False
          ent_ids.append(0)
          continue
        
        # Entity여부에 따라 id설정(entity - 1, 아니면 - 0)
        if flag:
          ent_ids.append(1)
        else:
          ent_ids.append(0)
      
      item["ent_ids"] = torch.tensor(ent_ids)


    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset, label_type,multi_task = None):
  """데이터와 라벨을 매칭시켜 데이터셋을 만들어주는 함수

  Args:
    dataset(DataFrame) : sentence, entity1, entity2에 대한 정보를 가지고 있는 DataFrame
    label_type(dice) : 관계명을 클래스의 번호와 매칭이 되어있는 딕셔너리
    multi_task(bool) : model을 multi task로써 만들지 단일 모델로써 만들건지 결정
  
  Returns:
    dataset(torch.utils.data.Dataset) : sentence, entity1, entity2, label이 들어가 있는 데이터 셋
  """

  label = []

  # multi_task model용으로 만들기 위한 list
  sentence_list = []
  entity01_list = []
  entity02_list = []
  entity_01_start_list = []
  entity_01_end_list = []
  entity_02_start_list = []
  entity_02_end_list = []

  
  # 단일 모델로 진행할지 혹은  멀티 모델로 진행할지
  if multi_task is None: # 주의! inference시에는 이것으로!
    for i in dataset[8]:
      if i == 'blind':
        label.append(100)
      else:
        label.append(label_type[i])
  elif multi_task == "model1":
    for i in dataset[8]:
      if i == 'blind':
        label.append(100)
      else:
        if i == "관계_없음":
          # print("관계 없음")
          label.append(0)
        else:
          # print("관계 있음")
          label.append(1)
  elif multi_task == "model2":
    for i, row in dataset.iterrows():
      label_class = label_type[row[8]]

      # 관계_없음을 제외한 클래스들만 학습(class 숫자를 1씩 감소)
      if label_class != 0:
        if i == 'blind':
          label.append(100)
        else:
          label.append(label_class - 1)

        sentence_list.append(row[1])
        entity01_list.append(row[2])
        entity02_list.append(row[5])
        entity_01_start_list.append(row[3])
        entity_01_end_list.append(row[4])
        entity_02_start_list.append(row[6])
        entity_02_end_list.append(row[7])

  # baseline
  # out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})

  # multitask model1 혹은 단일모델 수행
  if multi_task != "model2":
    # 원본용
    out_dataset = pd.DataFrame({
          'sentence':dataset[1],
          'entity_01':dataset[2],
          'entity_02':dataset[5],
          'label':label,
          'entity_01_start' : dataset[3],
          'entity_01_end' : dataset[4],
          'entity_02_start' : dataset[6],
          'entity_02_end' : dataset[7]
      })
  elif multi_task == "model2":
    # 관계 없음을 제외한 task를 수행할 모델 만들기 위한 데이터셋
    out_dataset = pd.DataFrame({
          'sentence':sentence_list,
          'entity_01':entity01_list,
          'entity_02':entity02_list,
          'label':label,
          'entity_01_start' : entity_01_start_list,
          'entity_01_end' : entity_01_end_list,
          'entity_02_start' : entity_02_start_list,
          'entity_02_end' : entity_02_end_list
      })

  # # 데이터 증강용
  # out_dataset = pd.DataFrame({
  #       'sentence':dataset[0],
  #       'entity_01':dataset[1],
  #       'entity_02':dataset[2],
  #       'label':dataset[3],
  #       # 'entity_01_start' : dataset[3],
  #       # 'entity_01_end' : dataset[4],
  #       # 'entity_02_start' : dataset[6],
  #       # 'entity_02_end' : dataset[7]
  #   })
  return out_dataset

def load_data(dataset_dir, multi_task = None):
  """ train데이터 셋을 불러오고 분류에 맞게 라벨링을 한 다음 초기 데이터 셋 구성

  Args:
    dataset_dir(str) : 데이터셋의 파일 경로
    multi_task(bool) : model을 multi task로써 만들지 단일 모델로써 만들건지 결정
  
  Returns:
    dataset(torch.utils.data.Dataset) : sentence, entity1, entity2, label이 들어가 있는 데이터 셋
  """
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type, multi_task)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer, ent_token, max_token_len, model_name):
  """문장에 대하여 토크나이징 된 결과를 반환해주는 함수입니다.
  
  Args:
    dataset(torch.utils.data.Dataset) : sentence, entity1, entity2, label이 들어가 있는 데이터 셋
    tokenizer
    ent_token(bool) : sentence에다가 ENT토큰 포함 여부 판별
    max_token_len(int) : 최대 input길이 설정

  Returns:
    tokenized_sentences(transformers.tokenization_utils_base.BatchEncoding) : 문장의 토큰화 된 결과
  """
  concat_entity = []
  concat_sentence = []
  ner = Pororo(task="ner", lang="ko") # 개체명 인식을 위한 pororo 객체 선언
  for e01, e02, e1s,e1e,e2s,e2e,sentence in tqdm(zip(
    dataset['entity_01'], 
    dataset['entity_02'],
    dataset['entity_01_start'],
    dataset['entity_01_end'],
    dataset['entity_02_start'],
    dataset['entity_02_end'],
    dataset['sentence']
    )):
  # for e01, e02,sentence in tqdm(zip(   # 외부데이터 정재하고 사용시 사용
  #   dataset['entity_01'], 
  #   dataset['entity_02'],
  #   dataset['sentence']
  #   )):
    if model_name != "xlm-roberta-large":
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)
    else:
      temp = ''
      temp = e01 + '</s>' + e02
      concat_entity.append(temp)

    if ent_token and model_name != "xlm-roberta-large":
      if e1s < e2s:
        sentence = sentence[:e1s] + '[ENT]' + sentence[e1s:e1e+1] + '[/ENT]' + sentence[e1e+1:e2s] + \
          '[ENT]' + sentence[e2s:e2e+1] + '[/ENT]'+ sentence[e2e+1:]
      elif e1s >= e2s:
          sentence = sentence[:e2s] + '[ENT]' + sentence[e2s:e2e+1] + '[/ENT]' + sentence[e2e+1:e1s] + \
          '[ENT]' + sentence[e1s:e1e+1] + '[/ENT]'+ sentence[e1e+1:]
      concat_sentence.append(sentence)
    elif ent_token and model_name == "xlm-roberta-large":
      ner_01 = ' ₩ ' + ner(e01)[0][1].lower() + ' ₩ '
      ner_02 =  ' ^ ' + ner(e02)[0][1].lower() + ' ^ '

      # 개체명 없는 버전
      if e1s < e2s:
        sentence = sentence[:e1s] + " # " + sentence[e1s:e1e+1] + " # " + sentence[e1e+1:e2s] + \
          " @ " +  sentence[e2s:e2e+1] + " @ "+ sentence[e2e+1:]
      elif e1s >= e2s:
          sentence = sentence[:e2s] + " @ " +  sentence[e2s:e2e+1] + " @ " + sentence[e2e+1:e1s] + \
          " # " + sentence[e1s:e1e+1] + " # " + sentence[e1e+1:]

      # 개체명 버전
      # if e1s < e2s:
      #   sentence = sentence[:e1s] + " # " + ner_01 + sentence[e1s:e1e+1] + " # " + sentence[e1e+1:e2s] + \
      #     " @ " + ner_02 + sentence[e2s:e2e+1] + " @ "+ sentence[e2e+1:]
      # elif e1s >= e2s:
      #     sentence = sentence[:e2s] + " @ " + ner_02 + sentence[e2s:e2e+1] + " @ " + sentence[e2e+1:e1s] + \
      #     " # " + ner_01 +sentence[e1s:e1e+1] + " # " + sentence[e1e+1:]
      concat_sentence.append(sentence)

  # 토큰 옵션 있는 경우
  if ent_token: 
    tokenized_sentences = tokenizer(
      concat_entity,
      concat_sentence,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=max_token_len,
      add_special_tokens=True,
      )
  # 토큰 옵션 없는 경우
  else: 
    tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=max_token_len,
      # max_length = 400,
      add_special_tokens=True,
      )

  return tokenized_sentences
