import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    
    def __init__(self, x, y, learning_rate: float, reg_lambda: float=0.01) -> None:
        self.x_cor = np.array(x)
        self.y_cor = np.array(y)

        if self.x_cor.ndim == 1:
            self.x_cor = self.x_cor.reshape(-1, 1)

        self.x_mean = np.mean(self.x_cor, axis=0)
        self.x_std = np.std(self.x_cor, axis=0)
        self.x_cor = (self.x_cor - self.x_mean) / self.x_std

        self.w = np.zeros(self.x_cor.shape[1])
        self.b = 0
        self.alpha = learning_rate
        self.reg_lambda = reg_lambda


    def lin_func(self, x):
        return np.dot(x, self.w) + self.b


    def MSE(self) -> float:
        error = self.lin_func(self.x_cor) - self.y_cor
        reg = (self.reg_lambda / (2 * self.x_cor.shape[0])) * np.dot(self.w, self.w)

        return (1 / (2 * len(self.x_cor))) * np.dot(error, error) + reg


    def gradient_descent(self) -> None:
        error = (self.lin_func(self.x_cor) - self.y_cor)
        reg = (self.reg_lambda / len(self.x_cor)) * self.w

        self.w -= self.alpha * ((1 / self.x_cor.shape[0]) * np.dot(self.x_cor.T, error) + reg)
        self.b -= self.alpha * (1 / self.x_cor.shape[0]) * np.sum(error)


    def denormalize_coefficients(self):
        temp_b = self.b - np.sum(self.w * self.x_mean / self.x_std)
        temp_w = self.w / self.x_std
        return temp_w, temp_b


    def fit(self, accuracy: float, max_iters: int = 1000000) -> None:
        iters = 0
        while self.MSE() > accuracy and iters < max_iters:
            self.gradient_descent()
            if iters % 100000 == 0:
                temp_w, temp_b = self.denormalize_coefficients()
                print(f"w: {temp_w}, b: {temp_b}")
            iters += 1

        self.w, self.b = self.denormalize_coefficients()


    def predict(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return self.lin_func(x)
