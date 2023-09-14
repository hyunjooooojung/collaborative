import numpy as np
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
import pickle
# from datetime import datetime


def preprocess_data():
    ''' 사용자, 악보 데이터 전처리 : 사용자-악보 행렬 생성, 인덱스 매핑 생성 '''

    # 데이터 불러오기 
    musics: pd.DataFrame = pd.read_csv('akbonara_musicSheet_info.csv', low_memory=False)
    orders: pd.DataFrame = pd.read_csv('akbonara_member_order_info_test.csv', low_memory=False)

    # 구매 여부를 나타내는 열 추가 (구매했으면 1, 그렇지 않으면 0)
    orders['구매여부'] = 1
    # print(orders.head(10))
    
    # 사용자-악보 행렬 생성
    user_music_matrix: pd.DataFrame = pd.pivot_table(orders, values='구매여부', index='mem_id', columns='cde_id', fill_value=0)
    # print(user_music_matrix)

    # 구매 여부 열 삭제 
    orders.drop('구매여부', axis=1, inplace=True)

    # 사용자 간의 유사도 계산 (코사인 유사도 사용)
    # user_similarity: np.ndarray = cosine_similarity(user_music_matrix)
    # print("코사인 유사도: ", user_similarity)

    return musics, orders, user_music_matrix


def recommend_music_for_user(mem_id: int, musics: pd.DataFrame, orders: pd.DataFrame):
    ''' 사용자와 악보 데이터의 임베딩 벡터로 추천 시스템 구현 '''
    
    ### SVD 모델 적용 후 악보 추천 ###

    filename = 'svd_model_v1.pkl'

    # pickle 파일에서 모델을 불러옴
    with open(filename, 'rb') as f:
        svd_model = pickle.load(f)
        print("저장된 svd 모델 :", svd_model)

    # 추천을 위한 사용자 선택 
    target_user_index = orders[orders['mem_id'] == mem_id].index[0]
    print(target_user_index)

    # 학습된 SVD 모델을 사용하여 악보를 추천
    user_factors = svd_model.user_factors
    item_factors = svd_model.item_factors
    pred_ratings = np.dot(user_factors[target_user_index], item_factors.T)
    print(pred_ratings)
    
    # 추천된 악보의 인덱스 및 예상 평점 (pred_ratings)을 정렬하여 상위 악보를 선택
    top_k = 10  # 추천할 악보 수 
    top_indices = pred_ratings.argsort()[-top_k:][::-1]

    # 추천된 악보 정보 출력
    recommended_music_info = musics.iloc[top_indices][['cde_id', 'title', 'artist', 'album', 'hit', 'price', 'part', 'genres', 'orderCnt']]
    print(recommended_music_info)

    # 특정 사용자의 구매 기록 필터링
    user_orders = orders[orders['mem_id'] == mem_id]
    
    # 특정 사용자가 구매한 악보의 cde_id 출력
    purchased_music_ids = user_orders['cde_id'].tolist()
    print(f"사용자 {mem_id}가 구매한 악보 id: {purchased_music_ids}")

    return recommended_music_info

