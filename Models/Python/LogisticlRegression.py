import numpy as np


class LogisticRegression:

    def __init__(self, x, y, learning_rate: float, reg_lambda: float=0.01) -> None:
        self.x_cor = np.array(x)
        self.y_cor = np.array(y)

        if self.x_cor.ndim == 1:
            self.x_cor = self.x_cor.reshape(-1, 1)

        self.w = np.zeros(self.x_cor.shape[1])
        self.b = 0
        self.alpha = learning_rate
        self.reg_lambda = reg_lambda


    def lin_func(self, x):
        return np.dot(x, self.w) + self.b
    

    def sigmoid_func(self, x):
        z = self.lin_func(x)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    def cost_func(self) -> float:
        predict = self.sigmoid_func(self.x_cor)
        predict = np.clip(predict, 1e-10, 1 - 1e-10)

        error = -self.y_cor * np.log(predict) - (1 - self.y_cor) * np.log(1 - predict)
        reg = (self.reg_lambda / (2 * self.x_cor.shape[0])) * np.dot(self.w, self.w)

        return (1 / self.x_cor.shape[0]) * np.sum(error) + reg


    def gradient_descent(self) -> None:
        error = self.sigmoid_func(self.x_cor) - self.y_cor
        m = self.x_cor.shape[0]
        reg = (self.reg_lambda / m) * self.w
        grad = (1 / m) * np.dot(self.x_cor.T, error)

        self.w -= self.alpha * (grad + reg)
        self.b -= self.alpha * (1 / m) * np.sum(error)


    def fit(self, accuracy: float, max_iters: int = 1000000) -> None:
        iters = 0
        while self.cost_func() > accuracy and iters < max_iters:
            self.gradient_descent()
            iters += 1


    def predict(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        probabilities = self.sigmoid_func(x)
        return round(probabilities, ndigits=4)
