import numpy as np


class LinearRegression:
    
    def __init__(self, x, y, learning_rate: float) -> None:
        self.w = 0
        self.b = 0
        self.alpha = learning_rate
        self.x_cor = np.array(x)
        self.y_cor = np.array(y)


    def lin_func(self, x) -> float:
        return self.w * x + self.b


    def MSE(self) -> float:
        error = self.lin_func(self.x_cor) - self.y_cor
        return (1 / (2 * len(self.x_cor))) * np.dot(error, error)


    def gradient_descent(self) -> None:
        temp_w = self.w
        temp_b = self.b
        error = (self.lin_func(self.x_cor) - self.y_cor)
        
        temp_w = self.w - self.alpha * (1 / len(self.x_cor)) * np.dot(error, self.x_cor)
        temp_b = self.b - self.alpha * (1 / len(self.x_cor)) * np.sum(error)
        
        self.w = temp_w
        self.b = temp_b


    def fit(self, accuracy: float, max_iters: int = 1000000) -> None:
        iters = 0
        while self.MSE() > accuracy and iters < max_iters:
            self.gradient_descent()
            iters += 1

        print(self.w, self.b)
