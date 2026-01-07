# 필요한 라이브러리들을 불러옵니다
import numpy as np                # 수치 계산용 라이브러리
import pandas as pd               # 데이터 처리 및 분석용 라이브러리
import matplotlib.pyplot as plt   # 그래프 그리기용 라이브러리
import seaborn as sns            # 통계 시각화 라이브러리
import os
import warnings
warnings.filterwarnings('ignore')

#데이터 경로 설정
current_dir = os.path.dirname(__file__)
test_path = os.path.join(current_dir, "test.csv")
train_path = os.path.join(current_dir, "train.csv")

# 훈련 데이터셋 로딩
print("데이터 로딩 중...")
train_dataset = pd.read_csv(train_path )

print(f"데이터 로딩 완료!")
print(f"데이터 형태: {train_dataset.shape[0]:,}행 {train_dataset.shape[1]}열")

# 데이터셋 미리보기
print("데이터 미리보기 (처음 5개 행):")
print(train_dataset.head())

print("\n데이터 미리보기 (마지막 5개 행):")
print(train_dataset.tail())

# 원본 데이터 복사 (원본 보존을 위해)
data = train_dataset.copy()

print("탐색적 데이터 분석 시작...")
print(f"데이터 크기: {data.shape[0]:,}행 {data.shape[1]}열")

# 데이터 타입 및 정보 확인
print("\n데이터 정보:")
print(data.info())


# 입력 변수(X)와 타겟 변수(y) 분리
X = data.drop(columns=['quality'])  # quality 컬럼 제거
y = data['quality']                 # quality만 선택
ids = data['id']                    # ID 컬럼 따로 저장

# X에서 Id 컬럼도 제거 (예측에 사용하지 않음)
X = X.drop(columns=['id'])
X = X.fillna(0) # 결측치 제거

print(f"입력 변수 (X) 형태: {X.shape}")
print(f"타겟 변수 (y) 형태: {y.shape}")
print(f"입력 변수 컬럼들: {list(X.columns)}")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.ensemble import VotingClassifier

print("필요한 라이브러리 import 완료!")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25,      # 25%를 테스트용으로 사용
    random_state=365,    # 재현 가능한 결과를 위한 시드값
    stratify=y          # 타겟 변수의 분포를 유지하며 분할
)

print(f"데이터 분할 완료!")
print(f"훈련 데이터 크기: {X_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")

# 분할 후 타겟 변수 분포 확인
print("\n분할 후 타겟 변수 분포:")
print("훈련 데이터:")
print(y_train.value_counts().sort_index())
print("\n테스트 데이터:")
print(y_test.value_counts().sort_index())

# 앙상블
model = VotingClassifier(
    estimators=[
        ('cat', CatBoostClassifier(iterations=900,learning_rate=0.045,depth=6,random_seed=6,verbose=100)),
        ('lgb', LGBMClassifier(n_estimators=900, learning_rate=0.03, num_leaves=25)),
        ('xgb', XGBClassifier(n_estimators=900, learning_rate=0.03, max_depth=6))
    ],
    voting='soft'
)
model.fit(X_train, y_train)

print("모델 훈련 완료!")

# 기본 성능 평가
print("모델 성능 평가:")

# 훈련 데이터와 테스트 데이터에 대한 정확도
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"훈련 정확도: {train_score:.4f} ({train_score*100:.2f}%)")
print(f"테스트 정확도: {test_score:.4f} ({test_score*100:.2f}%)")

# 상세 성능 분석
print("상세 성능 분석...")

# 예측 수행
y_pred = model.predict(X_test)

# 분류 리포트
print("상세 분류 리포트:")
print(classification_report(y_test, y_pred))

# 예측 정확도
accuracy = accuracy_score(y_test, y_pred)
print(f"\n테스트 데이터 예측 정확도: {accuracy:.4f}")

# 테스트 데이터 로딩 (실제 제출용 데이터)
try:
    test_data = pd.read_csv(test_path)
    print(f"제출용 테스트 데이터 로딩 완료!")
    print(f"제출용 테스트 데이터 크기: {test_data.shape}")
    
    # 테스트 데이터 미리보기
    print("\n테스트 데이터 미리보기:")
    print(test_data.head())
    
except FileNotFoundError:
    print("테스트 데이터 파일을 찾을 수 없습니다.")
    print("실제 환경에서는 테스트 데이터 경로를 확인해주세요.")
    # 임시로 훈련 데이터의 일부를 사용
    test_data = X.head(100).copy()
    print(f"임시 테스트 데이터 크기: {test_data.shape}")

    # 예측 실행
print("예측 실행 중...")

# test 데이터의 id를 먼저 저장
test_ids = test_data['id'].copy()

# 학습할 때 id 컬럼은 학습에서 제외했으므로 test에서도 제외
test_data = test_data.drop(columns=['id'])
test_data = test_data.fillna(0) # 결측치 제거

# 예측하는 코드
predictions = model.predict(test_data)

# 예측값이 2차원이라면 1차원으로 펴주기
if predictions.ndim > 1:
    predictions = predictions.flatten()

# 결과 DataFrame 생성
results_df = pd.DataFrame({
    'id': test_ids,
    'quality': predictions
})

print(f"예측 완료!")
print(f"총 {len(predictions)}개 샘플 예측")
print(f"예측값 범위: {predictions.min()} ~ {predictions.max()}")

# 결과를 DataFrame으로 정리 및 저장
print("결과 저장 중...")

# 결과 DataFrame 생성 (test_ids는 이전 셀에서 저장한 원본 id 사용)
results_df = pd.DataFrame({
    'id': test_ids,
    'quality': predictions
})

# CSV 파일로 저장
output_filename = 'submission.csv'
results_df.to_csv(output_filename, index=False)

print(f"결과가 '{output_filename}' 파일로 저장되었습니다.")
print(f"저장된 결과: {len(results_df)}행 {len(results_df.columns)}열")

print("\n저장된 결과 미리보기:")
print(results_df.head(10))

print("\n저장된 결과 마지막 부분:")
print(results_df.tail(5))

