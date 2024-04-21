import numpy as np



class Rastrigin:
    def __init__(self, dim):
        self.dim = dim

    def evaluate(self, x):
        return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)


class Rosenbrock:
    def __init__(self, dim):
        self.dim = dim

    def evaluate(self, x):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)


class Ackley:
    def __init__(self, dim):
        self.dim = dim

    def evaluate(self, x):
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / self.dim)) - \
               np.exp(np.sum(np.cos(2 * np.pi * x)) / self.dim) + 20 + np.exp(1)


class Esfera:
    def __init__(self, dim):
        self.dim = dim

    def evaluate(self, x):
        return np.sum(x**2)
