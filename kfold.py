import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    # 학습 데이터는 train.csv에 있음
    df = pd.read_csv('train.csv')

    # kfold라는 새로운 열 생성하고 -1로 채움
    df['kfold'] = -1

    # 데이터의 행을 랜덤하게 섞음
    df = df.sample(frac=1).reset_index(drop=True)
    
    # kfold 클래스 초기화
    kf = model_selection.KFold(n_splits=5)

    # kfold열을 폴드 아이디로 설정
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    # 데이터를 kfold 열과 함께 새로운 csv 파일로 저장
    df.to_csv('train_folds.csv', index=False)