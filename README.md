# 문장 내 개체간 관계 추출 Project(Boostcamp P stage2)
본프로젝트는 NAVER AI BoostCamp에서 개최한 competition입니다

## 최종 결과
**`Private accuracy`** 80.4000%

**`Rank`** 28/136

## 대회 목표
문장, 엔티티, 관계에 대한 정보를 통해, 문장과 엔티티 사이의 관계를 추론하여 분류

## 문제 해결법

- 해당 문제를 풀기 위해 어떻게 접근하고, 구현하고, 최종적으로 사용한 솔루션에 대해서는 [report](https://www.notion.so/P-stage2-KLUE-Wrap-UP-Report-cbc9e97009004e6a89d3e87442a3e575)에서 확인할 수 있습니다

- 위 Report는 개인의 회고도 포함하고 있습니다.


## 📜 최종 모델 및 파라미터
   1. model
        1.  xlm-roberta-large
   2. Learning Rate: 5e-5
   3. Optimizer : Adam
   4. Loss : CE
   5. Epoch: 20
   6. Scheduler : linear
   7. batch_size : 16

## 🌟 문제 해결 측면
### **[EDA]**
1. 42개의 클래스가 어떠한 식으로 분포되어 있는지 시각화 - pieplot

2. 대분류(관계 없음, 단체, 인물)의 데이터 분포도 파악 - pieplot

3. 인물, 단체 클래스에 대해서만 데이터 분포도 파악 = pieplot

4. 문장의 길이, 모델별 다양한 pretrain된 토크나이저를 문장에 대하여 사용 후 길이에 대한 분석 - boxplot

5. 모델별 테스트 데이터를 기준 UNK 토큰 개수 분포도 파악 - boxplot

### **[Text Augmentation]**
1.EDA(Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks)

    (1) 문장에 단어를 유의어로 교체(SR)
    (2) 문장에 단어를 랜덤 삽입(RI)
    (3) 문장의 단어를 랜덤 교체(RS)
    (4) 문장의 단어를 랜덤 삭제(RD)
 
2. BackTranslation

### **[Text Preprocessing]**
1. 데이터 셋에 대하여 한국어 말고도 한자나 괄호같은 다양한 기호나 문자들이 들어가 있어 전처리 후 학습을 진행

2. xlm-roberta-large 모델 사용시 엔티티에 해당하는 토큰을 스페셜 토큰으로 감싸줌
( entity가 들어있는 문장 앞에 개체명을 넣고 개체명은 '₩', '^' 의 토큰으로 감싸고 entity는 "#", "@"로 감쌈)

### **[외부데이터 가공]**
1. 관계 없음 클래스를 제외한 클래스에 대해서 외부 데이터를 가공해서 데이터셋에 추가 -> 부족한 validation set으로 사용

### **[model 튜닝]**
1. Entity Embedding 추가
<img src = "https://user-images.githubusercontent.com/51118441/120637631-1a520780-c4aa-11eb-9c4d-a889e1fe9fdb.png" width="70%">

2. last hidden state사용<br>
ver1(추가 층 추가)<br>
<img src = "https://user-images.githubusercontent.com/51118441/120637708-32298b80-c4aa-11eb-81b4-e45c99d8b8ba.png" width="70%"><br>
ver2(pooled output에 entity의 hidden state 값을 합쳐 평균을 구해 새로운 pooled output 도출)<br>
<img src = "https://user-images.githubusercontent.com/51118441/120637727-35247c00-c4aa-11eb-89a7-b48b7d9633c3.png" width="70%">

### **[multi task]**
1. EDA를 하였을 시 관계_없음과 나머지 클래스의 비율이 거의 1:1로 가까워 이를 2가지 task로 나눠 모델 학습을 하고 분류를 진행
<img src = "https://user-images.githubusercontent.com/51118441/120637813-4f5e5a00-c4aa-11eb-97bf-845e7b241ce1.png" width="50%">

### **[이외의 시도]**
1. 데이터 셋의 max length길이 분포를 구하고 모델에 들어가는 문장의 max length길이 조정

2. 클래스 불균형 해소를 위한 focalloss사용

3. Ensemble수행
