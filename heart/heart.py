# 파이썬 패키지 수입
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

import torch
from torch import nn
from sklearn.model_selection import train_test_split # 학습용, 평가용 분류
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score # 정확도 계산

# 하이퍼 파라미터
INPUT_DIM = 13      # 입력
MY_HIDDEN = 1000    # 은닉층
MY_EPOCH = 1000     # 학습 횟수

# 실습 1 : seed가 없으면 매번 실행 때마다 결과가 달라진다. =>  f1 점수(정확도)

# 추가 옵션: 임의의 수 생성 및 지정
pd.set_option('display.max_columns', None)
torch.manual_seed(111)      # 임의의 수 생성
import numpy as np
np.random.seed(111)

############## 데이터 준비 ##############

# 데이터 파일 읽기
# 결과는 pandas의 데이터 프레임 형식
raw = pd.read_csv('heart.csv')

# 데이터 원본 출력
print('원본 데이터 샘플 10개')
print(raw.head(10))
print('원본 데이터 통계')
print(raw.describe())

# 데이터를 입력과 출력으로 분리
X_data = raw.drop('target', axis=1)
Y_data = raw['target']
names = X_data.columns      # 이름 저장
print(names)

# 데이터를 학습용과 평가용으로 분리
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)

# 최종 데이터 모양 : 4분할 후 데이터 모양 확인
print('\n학습용 입력 데이터 모양:', X_train.shape)   # 2차원 행렬
print('학습용 출력 데이터 모양:', Y_train.shape)     # 1차원 행렬 or 1차원 벡터
print('학습용 입력 데이터 모양:', X_test.shape)      # 2차원 행렬
print('학습용 출력 데이터 모양:', Y_test.shape)      # 1차원 행렬 or 1차원 벡터

# 입력 데이터 Z-점수 정규화
# 결과는 numpy의 n-차원 행렬 형식
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print('\n전환 전:', type(X_train))

# numpy에서 pandas로 변환
# header 정보 복구 필요
X_train = pd.DataFrame(X_train, columns=names)
X_test = pd.DataFrame(X_test, columns=names)
print('전환 후:', type(X_train))

# 정규화 된 학습용 데이터 출력
print('\n정규화 된 학습용 데이터 샘플 10개')
print(X_train.head(10))
print('\n정규화 된 학습용 데이토 통계')
print(X_train.describe()) # 13개의 평균값, 표준편차

# 학습용 데이터 상자 그림
sns.set(font_scale=2)
# sns.boxplot(data=X_train, palette="colorblind")
# plt.show()

############## 인공 신경망 구현 ##############

# 실습 3 : 은닉층 추가 => 가중치, 학습 시간, 정확도 증가
# nn.Linear(MY_HIDDEN, 5000),         # 2 & 3번째 은닉층
# nn.Tanh(),
# nn.Linear(5000, 1),
# 실습 4 : Tanh() -> ReLU() => 가중치 변화X, 정확도 & 학습시간 증가

# 파이토치 DNN을 Sequential 모델로 구현
model = nn.Sequential(
    nn.Linear(INPUT_DIM, MY_HIDDEN),    # 입력층 & 1번째 은닉층
    nn.Tanh(),
    nn.Linear(MY_HIDDEN, MY_HIDDEN),    # 1 & 2번째 은닉층
    nn.Tanh(),
    nn.Linear(MY_HIDDEN, 1),            # 2번째 은닉층
    nn.Sigmoid()
)

print('\nDNN 요약')
print(model)

# 총 파라미터 수 계산
total = sum(p.numel() for p in model.parameters())
print('총 파라미터 수: {:,}'.format(total))

# 실습 2 : optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 실습 2 : optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01) => 정확도 감소, 학습 시간 증가

# 최적화 함수와 손실 함수 지정
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 학습용 데이터 전환
# pandas dataframe에서 pytorch 텐서로
print('\n전환 전:', type(X_train))
X_train = torch.tensor(X_train.values).float()
Y_train = torch.tensor(Y_train.values).float()
print('전환 후:', type(X_train))

############## 인공 신경망 학습 ##############

# DNN 학슴
begin = time()
print('\nDNN 학습 시작')

# 순방향 계산
for epoch in range(MY_EPOCH):
    output = model(X_train)
    # print(X_train.shape)
    # print(output.shape)

    # 출력값 차원을 (212, 1)에서 (212, ) 로 조정 : 2차원 -> 1차원
    output = torch.squeeze(output)
    # print('심장병 판결 결과:', output.shape)

    # 손실값 계산
    loss = criterion(output, Y_train)
    # print('학습용 출력 데이터:', Y_train.shape)

    # 손실값 출력
    if(epoch % 10 == 0):
        print('에포크: {:2}'.format(epoch),
              '손실: {:.3f}'.format(loss.item()))

    # 역전파 알고리즘으로 가중치 보정
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
end = time()

print('최종 학습 시간: {:.1f}초'.format(end - begin))

############## 인공 신경망 평가 ##############

# 평가용 데이터 전환
# pandas dataframe에서 pytorch 텐서로
X_test = torch.tensor(X_test.values).float()

# DNN을 추측, 가중치 관련 계산 불필요
with torch.no_grad():
    pred = model(X_test)

print(pred.flatten())

# 추측 결과 tensor를 numpy로 전환
pred = pred.numpy()

# 확률을 이진수로 전환 후, F1 점수 계산
pred = (pred > 0.5)
print('추측값:', pred.flatten()) # 출력 모양을 깔끔하게 해줌
print('정답:', Y_train.flatten())

f1 = f1_score(Y_test, pred)
print('\n최종 정확도 (F1 점수): {:.3f}'.format(f1))

