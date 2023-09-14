import pandas as pd
import numpy as np
import argparse
from model import SVD, SGD
from util import make_sparse_matrix
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    ''' 필요한 사용자, 악보 데이터를 로드하고 사용자별로 구매내역을 저장하는 함수 '''

    # 데이터 불러오기
    musics = pd.read_csv('akbonara_musicSheet_info_small.csv')
    print(musics.head(5))
    orders = pd.read_csv('akbonara_member_order_info_small.csv')
    print(orders.head(5))

    # 사용자별로 구매한 악보의 ID를 리스트로 저장
    user_purchase_history = orders.groupby('mem_id')['cde_id'].apply(list).reset_index(name='purchase_history')
    print(user_purchase_history.head(10))

    return musics, orders, user_purchase_history

def preprocess_data(musics: pd.DataFrame, orders: pd.DataFrame, user_purchase_history: pd.DataFrame):
    ''' 사용자, 악보 데이터 전처리 : 사용자-악보 행렬 생성, 인덱스 매핑 생성 '''

    # 사용자와 악보 데이터의 인덱스 매핑
    user_to_index: dict = {user: index for index, user in enumerate(user_purchase_history['mem_id'])}
    music_to_index: dict = {music: index for index, music in enumerate(musics['cde_id'])}

    # 사용자-악보 행렬 생성
    user_indices: list = []
    music_indices: list = []
    ratings: list = []
    for _, row in orders.iterrows():
        user_idx = user_to_index[row['mem_id']]
        cde_id = row['cde_id']
        if cde_id in music_to_index:
            music_idx = music_to_index[cde_id]
            user_indices.append(user_idx)
            music_indices.append(music_idx)
            ratings.append(1)  # 모든 구매에 대해 평점 1로 통일

    return user_to_index, music_to_index, user_indices, music_indices, ratings

def recommend_music_for_user(mem_id, user_factors, music_factors, music_to_index, top_k=5):
    ''' 사용자와 악보 데이터의 임베딩 벡터로 추천 시스템 구현 '''

    # 특정 사용자의 임베딩 벡터 추출
    user_idx = user_to_index[mem_id]
    user_embedding_weights = user_factors[user_idx]

    # 코사인 유사도 계산
    similarities = cosine_similarity([user_embedding_weights], music_factors)[0]

    # 코사인 유사도가 높은 순으로 추천 악보를 생성.
    recommended_music_indices = similarities.argsort()[::-1]

    # 추천 악보 개수 설정
    recommended_music_ids = [list(music_to_index.keys())[list(music_to_index.values()).index(music_idx)] for music_idx in recommended_music_indices[:top_k]]
    print(recommended_music_ids)
    return recommended_music_ids


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


def main(config):
    musics, orders, user_purchase_history = load_data()
    sparse_matrix, test_set = make_sparse_matrix(user_purchase_history)
    print(sparse_matrix)
    print("Sparse Matrix shape:", sparse_matrix.shape)
    print("Test set length:", len(test_set))
    
    print(config.k)
    if config.svd:
        print("엥?:")
        trainer = SVD(sparse_matrix, config.k)
    # elif config.sgd:
    #     trainer = SGD(                   
    #         sparse_matrix,
    #         config.k,
    #         config.lr,
    #         config.beta,
    #         config.n_epochs
    #     )
    else:
        raise RuntimeError('Algorithm No Selected')

    trainer.train()
    print("train RMSE:", trainer.evaluate())
    print("test RMSE:", trainer.test_evaluate(test_set))


if __name__ == '__main__':
    config = define_argparser()
    musics, orders, user_purchase_history = load_data()
    user_to_index, music_to_index, user_indices, music_indices, ratings = preprocess_data(musics, orders, user_purchase_history)
    
    # 사용자-악보 행렬 생성
    num_users = len(user_to_index)
    num_music = len(music_to_index)
    sparse_matrix = np.zeros((num_users, num_music))
    
    for user_idx, music_idx in zip(user_indices, music_indices):
        sparse_matrix[user_idx, music_idx] = 1  # 구매한 악보에 대해 평점 1로 설정
    
    print(sparse_matrix)
    # 추천 시스템 모델 학습 및 평가
    trainer = SVD(sparse_matrix, config.k)
    trainer.train()
    print("train RMSE:", trainer.evaluate())

    # 테스트 데이터로 평가
    test_set = [(user_idx, music_idx, 1) for user_idx, music_idx in zip(user_indices, music_indices)]
    print("test RMSE:", trainer.test_evaluate(test_set))


    # 특정 사용자에게 악보 추천
mem_id = 1003323  # 추천을 받을 사용자의 ID를 여기에 입력하세요.
recommended_music = recommend_music_for_user(mem_id, trainer.user_factors, trainer.item_factors, music_to_index, top_k=5)
print("Recommended Music IDs:", recommended_music)
