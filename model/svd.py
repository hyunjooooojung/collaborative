import numpy as np
from sklearn.metrics import mean_squared_error

class SVD:

    def __init__(self, sparse_matrix, K):
        """
        Arguments
        - sparse_matrix : user-item rating matrix
        - K (int)       : number of latent dimensions
        """
        self.sparse_matrix = sparse_matrix
        self.K = K
        self.init_sparse_matrix()

    def init_sparse_matrix(self):
        self.train_matrix = self.sparse_matrix.copy()  # Create a copy of the sparse_matrix
        self.train_matrix[np.isnan(self.train_matrix)] = 0  # Replace NaN values with 0

    def train(self):
        print("Factorizing...")
        item_factors, user_factors = self.get_svd(self.train_matrix, self.K)
        self.item_factors = item_factors
        self.user_factors = user_factors
        self.pred_matrix = np.matmul(item_factors, user_factors.T)

    @staticmethod
    def get_svd(sparse_matrix, K):
        U, s, VT = np.linalg.svd(sparse_matrix.transpose())
        U = U[:, :K]
        s = np.diag(s[:K])
        VT = VT[:K, :]

        item_factors = np.transpose(np.matmul(s, VT))
        user_factors = U

        return item_factors, user_factors

    def evaluate(self):
        # Compute RMSE between self.sparse_matrix and self.pred_matrix
        idx, jdx = self.sparse_matrix.nonzero()
        ys = self.sparse_matrix[idx, jdx]
        preds = self.pred_matrix[idx, jdx]
        error = mean_squared_error(ys, preds, squared=False)
        return error

    def test_evaluate(self, test_set):
        # Compute RMSE for the test set
        ys, preds = [], []
        for i, j, rating in test_set:
            ys.append(rating)
            preds.append(self.pred_matrix[i, j])
        error = mean_squared_error(ys, preds, squared=False)
        return error
