from fastapi import FastAPI, Query
import uvicorn
import argparse
import time
import pickle
import pandas as pd
from model import SVD
from util import preprocess_data, recommend_music_for_user

app = FastAPI()


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--k',
        type=int,
        help='latent factor size.'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=200,
        help='num of Iterations'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate.'
    )
    p.add_argument(
        '--beta',
        type=float,
        default=0.01,
        help='regularization parameter.'
    )
    p.add_argument(
        '--svd',
        action='store_true',
        help='Use SVD Algorithm.'
    )
    p.add_argument(
        '--sgd',
        action='store_true',
        help='Use SGD Algorithm.'
    )
    return p.parse_args()


@app.get("/")
def health_check():
    return {"message": "fastAPI is running"}


@app.get("/api/training")
def training(
    k: int = Query(10, description="latent factor size"),
    svd: bool = Query(True, description="Use SVD Algorithm")
    ):
    ''' SVD 모델로 학습 진행 후 pickle 파일로 모델 저장 '''
    
    # 학습 시작
    start = time.time()

    # 데이터 로드
    musics, orders, user_music_matrix = preprocess_data()

    # SVD 모델 초기화 및 적용
    svd_model = SVD(user_music_matrix.values, k)
    svd_model.train()

    # 모델을 pickle 파일로 저장
    filename = f'svd_model_v1.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(svd_model, f)

    # 학습 완료
    end = time.time()
    print("학습 소요시간:", (end - start))

    return {"message": "모델 학습 후 파일 저장 완료", "모델 파일 이름": filename}


@app.get("/api/recommend/")
def recommend_music(mem_id: int = Query(..., description="사용자 id 입력")):
    ''' Pickle 파일에 저장된 학습 시켜놓은 모델을 불러와 추천 악보 생성 '''

    # 추천 시작
    start = time.time()

    # utils.py 함수 호출
    musics, orders, user_music_matrix = preprocess_data()
    recommended_music_info = recommend_music_for_user(mem_id, musics, orders)
    
    # 추천 완료
    end = time.time()
    print("추천 소요시간:" , (end - start))

    return recommended_music_info.to_dict(orient='records')
