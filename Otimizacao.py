import numpy as np

class OptimizationProblem:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def objective_function(self, x):
        raise NotImplementedError("Subclasses must implement objective_function method")

    def random_solution(self):
        return self.lower_bound + np.random.rand(self.dimension) * (self.upper_bound - self.lower_bound)



class Rastrigin(OptimizationProblem):
    def objective_function(self, x):
        A = 10
        return A * self.dimension + np.sum(x**2 - A * np.cos(2 * np.pi * x))



class Rosenbrock(OptimizationProblem):
    def objective_function(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)



class Ackley(OptimizationProblem):
    def objective_function(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        return -20 * np.exp(-b * np.sqrt(np.sum(x**2) / self.dimension)) - np.exp(np.sum(np.cos(c * x)) / self.dimension) + a + np.exp(1)

class Sphere(OptimizationProblem):
    def objective_function(self, x):
        return np.sum(x**2)