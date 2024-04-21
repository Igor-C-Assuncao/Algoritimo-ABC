import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline





class ABC:
    def __init__(self, func, n_agents, n_iter, dim, limit):
        self.func = func
        self.n_agents = n_agents
        self.n_iter = n_iter
        self.dim = dim
        self.limit = limit
        self.agents = np.random.uniform(low=-self.limit, high=self.limit, size=(self.n_agents, self.dim))
        self.fitness = np.zeros(self.n_agents)

    def initialize_population(self):
        
        self.agents = np.random.uniform(low=-self.limit, high=self.limit, size=(self.n_agents, self.dim))

    def employed_bees_phase(self):
        for i in range(len(self.agents)):
            new_agent = self.explore(self.agents[i], i)
            new_fitness = self.func(new_agent)
            if new_fitness < self.fitness[i]:
                self.agents[i] = new_agent
                self.fitness[i] = new_fitness

    def onlooker_bees_phase(self):
        total_fitness = np.sum(self.fitness)
        probabilities = self.fitness / total_fitness
        for i in range(self.n_agents):
            selected_index = np.random.choice(self.n_agents, p=probabilities)
            new_agent = self.explore(self.agents[selected_index], i)
            new_fitness = self.func(new_agent)
            if new_fitness < self.fitness[i]:
                self.agents[i] = new_agent
                self.fitness[i] = new_fitness

    def scout_bees_phase(self):
        for i in range(len(self.agents)):
            if np.random.uniform() < 0.1:  
                self.agents[i] = np.random.uniform(low=-self.limit, high=self.limit, size=self.dim)
                self.fitness[i] = self.func(self.agents[i])

    def explore(self, agent, idx):
        selected_index = np.random.choice(np.delete(np.arange(self.n_agents), idx))  
        phi = np.random.uniform(low=-1, high=1, size=self.dim)
        new_agent = agent + phi * (agent - self.agents[selected_index])
        new_agent = np.clip(new_agent, -self.limit, self.limit)
        return new_agent

    def optimize(self):
        best_solution = None
        best_fitness = np.inf

        for _ in range(self.n_iter):
            self.employed_bees_phase()
            self.onlooker_bees_phase()
            self.scout_bees_phase()

            min_index = np.argmin(self.fitness)
            if self.fitness[min_index] < best_fitness:
                best_solution = self.agents[min_index].copy()
                best_fitness = self.fitness[min_index]

        return best_solution, best_fitness

   





  