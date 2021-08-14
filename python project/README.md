## 코드에 대한 전반적인 설명
dataloader.load_data.py : 데이터 로딩 및 데이터 셋에 대한 구축 파일 입니다.

RE_Dataset : 학습에 사용할 데이터 셋을 정의한 클래스입니다.

preprocessing_dataset : 데이터와 라벨을 매칭시켜 데이터셋을 만들어주는 함수입니다.

load_data : train데이터 셋을 불러오고 분류에 맞게 라벨링을 한 다음 초기 데이터셋을 구성하는 함수입니다.

tokenized_dataset : 문장에 대하여 토크나이징 된 결과를 반환해주는 함수입니다.

---

evaluation.evaluation.py : 평가를 위한 파일입니다.

---

inference.py : 생성된 모델을 활용하여 평가 파일을 만들기 뤼한 파일입니다.

inference : 단일 모델 추론을 위한 함수입니다.

inference_multitask : 다중 모델 추론을 위한 함수입니다.

load_test_dataset : 추론 데이터셋을 파일에서 불러오기 위한 함수입니다.

CFGInference : 추론시 필요 파라미터를 관리해주는 클래스입니다.

---

models.model_tunning_example.py : huggingface내부에서 모델 튜닝을 해본 파일입니다.

models.models.py : huggingface에서 제동되는 모델을 이용하여 외부적으로 모델 튜닝을 해 본 파일입니다.

---

train.py : model훈련을 위한 파일입니다.

__get_logger : 로깅을 하기 위한 객체를 반환해주는 함수입니다.

seed_everything : seed값 고정을 하기 위한 함수입니다.

ParameterError : 필요 파라미터가 들어오지 않았을 때 예외를 반환해주는 클래스입니다.

compute_metrics : trainer의 평가 함수를 등록해 주기 위한 함수입니다.

train : 모델 훈련을 위한 함수입니다.

---

uilt.loss.py : 다양한 loss 함수를 정의한 파일입니다.

