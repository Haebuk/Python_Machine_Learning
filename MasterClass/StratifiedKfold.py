import pandas as pd
from sklearn import model_selection
# 학습 데이터는 train.csv에 있음
df = pd.read_csv('train.csv')

# kfold라는 새로운 열을 생성하고 -1로 채움
df['kfold'] = -1

# 데이터 랜덤하게 섞기
df = df.sample(frac=1).reset_index(drop=True)

# 타겟 변수 가져오기
y = df.target.values

# StratifiedKFold 클래스 초기화
kf = model_selection.StratifiedKFold(n_splits=5)

# kfold 열을 폴드 아이디로 설정
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

# 데이터를 kfold열과 함께 새로운 csv파일로 저장
df.to_csv('train_folds.csv', index=False)