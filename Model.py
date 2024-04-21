from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ABCOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self, algorithm, problem, n_agents, n_iter, dim, limit):
        self.algorithm = algorithm
        self.problem = problem
        self.n_agents = n_agents
        self.n_iter = n_iter
        self.dim = dim
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        func = self.problem(dim=self.dim).evaluate
        abc = self.algorithm(func=func, n_agents=self.n_agents, n_iter=self.n_iter, dim=self.dim, limit=self.limit)
        best_solution, best_fitness = abc.optimize()
        return best_solution, best_fitness


