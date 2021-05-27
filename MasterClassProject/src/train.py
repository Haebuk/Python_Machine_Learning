# -*- coding: utf-8 -*-
import joblib
import pandas as pd
from sklearn import metrics, tree

def run(fold):
    # 폴드가 정의되어있는 학습 데이터를 로드
    df = pd.read_csv('../input/mnist_train_folds.csv')

    # 학습 데이터는 kfold 열의 값이 제공된 fold와 다른 샘플
    # 인덱스를 리셋하였음을 유의
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # 검증 데이터는 kfold 열의 값이 제공된 fold와 같은 샘플
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 타겟 열을 제거하고 남은 피쳐를 넘파이 행렬로 변환
    # 타겟 변수는 데이터프레임의 label 열
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    # 검증 데이터도 동일하게 적용
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.labels.values

    # sklearn의 의사결정트리 모델 초기화
    clf = tree.DecisionTreeClassifier()
    
    # 학습데이터로 모델 적합
    clf.fit(x_train, y_train)

    # 검증 데이터의 예측 값 생성
    preds = clf.predict(x_valid)

    # 정확도 계산 및 출력
    accuracy = metrics.accuracy_score(y_valid, preds)
    print("Fold={}, Accuracy={}".format(fold, accuracy))

    # 모델 저장
    joblib.dump(clf, "../models/dt_{}.bin".format(fold))

if __name__ == '__main__':
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)