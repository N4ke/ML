import numpy as np


class LinearRegression:
    
    def __init__(self, x, y, learning_rate: float) -> None:
        self.x_cor = np.array(x)
        self.y_cor = np.array(y)

        if self.x_cor.ndim == 1:
            self.x_cor = self.x_cor.reshape(-1, 1)
        
        self.x_cor = (self.x_cor - np.mean(self.x_cor, axis=0)) / np.std(self.x_cor, axis=0)
        self.y_cor = (self.y_cor - np.mean(self.y_cor, axis=0)) / np.std(self.y_cor, axis=0)

        self.w = np.zeros(self.x_cor.shape[1])
        self.b = 0
        self.alpha = learning_rate


    def lin_func(self, x) -> float:
        return np.dot(x, self.w) + self.b


    def MSE(self) -> float:
        error = self.lin_func(self.x_cor) - self.y_cor
        return (1 / (2 * len(self.x_cor))) * np.dot(error, error)


    def gradient_descent(self) -> None:
        error = (self.lin_func(self.x_cor) - self.y_cor)
        
        self.w -= self.alpha * (1 / len(self.x_cor)) * np.dot(self.x_cor.T, error)
        self.b -= self.alpha * (1 / len(self.x_cor)) * np.sum(error)


    def fit(self, accuracy: float, max_iters: int = 1000000) -> np.array:
        iters = 0
        while self.MSE() > accuracy and iters < max_iters:
            self.gradient_descent()
            iters += 1

        return self.w, self.b
