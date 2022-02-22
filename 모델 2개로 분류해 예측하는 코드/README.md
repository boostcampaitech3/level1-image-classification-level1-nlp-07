# More Coffee 팀 훈련 및 결과 제출


## MZ 인공지능 해커톤 대회
#### ‘AI  장치용 STT를 위한 의도 분류 (AI- NLU with STT(Speech To Text))’


## Task
#### 785개 의도 labels를 예측하는 Text Classification


## 사용 모델
#### Bert 기반 모델 중 ** KoELECTRA **를 선택했습니다.


## “KoELECTRA” 모델 소개
#### 여러 버전 중 KoELECTRA-Small-v3 버전을 사용했습니다.
#### Vocab Size 3만 5천개로 Pretrained 된 모델입니다.
##### reference : https://github.com/monologg/KoELECTRA


## 코드 진행
#### Class Imbalance 완화를 위해,  모델  두 개를 연결하여 앙상블하였습니다.
#### Class 수가 적은 60개 label 기준으로 해당 Class 데이터를 ‘others’로 치환, 데이터를 슬라이싱 하여 모델1 훈련 및 예측 후, 나머지 ‘others’에 해당하는 데이터로 모델2 훈련 후 예측하였습니다.

*Training 과정
- 첫번째 모델 1.선언 -> 2.training -> 3.저장
model = ☆★☆★☆
model.fit(padded_first_tr, label_first_tr, validation_data = [padded_first_dev, label_first_dev], ☆★☆★☆)
model.save(☆★☆my_first_model☆★☆) 

- 두번째 모델 1.선언 -> 2.training -> 3.저장
model = ☆★☆★☆ # model.fit(padded_second_tr, label_second_tr, validation_data = [padded_second_dev, label_second_dev], ☆★☆★☆)
model.save(☆★☆my_first_model☆★☆)

#### 훈련 성능은 모델1+ 모델2 앙상블 결과, dev set 기준 ACC 67.3% 나왔습니다.

——————————————————————————————————

# Docker 실행 명령어 순서

#### 윈도우 10 home 에서 작성하였습니다. 
#### Cmd (명령어 프롬프트) 에서 하시면 잘 돌아갑니다.

1.   이미지 불러오기 / docker pull tensorflow/tensorflow:2.4.0
2.   도커실행 / docker run -dit --name more_coffee3 tensorflow/tensorflow:2.4.0 bash
3.   도커진입 / docker exec –i –t more_coffee3 bin/bash
4.   apt-get update
5.   apt-get upgrade
6.   apt-get install vim
7.   exit
8.   host의 파일 경로입니다 --> 이걸 컨테이너로 copy  
docker cp C:\submission/class60/. more_coffee3:/
9.   docker exec -i -t more_coffee3 bin/bash 로 진입
10.   ls 쳐서 잘 들어가있는지 확인
11.   있으면 exit 로 빠져나옴
12.   docker commit more_coffee3 mc_3
13.   docker save -o mc_3.tar.gz mc_3
14.   (저장된 mc_3 의 위치는 개인마다 다르므로 검색에서 이름쳐서 폴더확인)
15.   docker exec -i -t more_coffee3 bin/bash
16.   pip install –-no–cache-dir -r requirements.txt.txt
17.   python prediction.py —input_text=test.txt —output_text = result.txt


# Docker 제출 내용

## 1. prediction.py
#### 평가시 inference 예측을 위한 코드입니다.
#### $python prediction.py —input_text=test.txt —output_text = result.txt 실행 결과, result.txt를 생성합니다.

## 2. model_save 폴더
#### 훈련된 모델 파일 및 훈련 스크립트 입니다.

## 3. Requirement.txt.txt
#### Requirement.txt 관련 주의할 점으로, torch 설치할 때 $pip install —no-cache-dir -r requirements.txt.txt 명령어로 실행해주어야 합니다. (Memory Error 문제 예방을 위함)


