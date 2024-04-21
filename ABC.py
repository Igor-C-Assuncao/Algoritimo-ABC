import numpy as np

class ABC:
    def __init__(self, func, n_agents, n_iter, dim, limit):
        self.func = func
        self.n_agents = n_agents
        self.n_iter = n_iter
        self.dim = dim
        self.limit = limit

    def optimize(self):
        best_solution = None
        best_fitness = np.inf
        agents = np.random.uniform(low=-self.limit, high=self.limit, size=(self.n_agents, self.dim))

        for _ in range(self.n_iter):
            for i, agent in enumerate(agents):
                new_agent = self.explore(agent, agents)
                fitness = self.func(new_agent)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = new_agent.copy()
                if fitness < self.func(agent):
                    agents[i] = new_agent.copy()
        
        return best_solution, best_fitness

    def explore(self, agent, agents):
        selected_agents = agents[np.random.choice(len(agents), 3, replace=False)]
        phi = np.random.uniform(low=-1, high=1, size=self.dim)
        new_agent = agent + phi * (agent - selected_agents.mean(axis=0))
        new_agent = np.clip(new_agent, -self.limit, self.limit)
        return new_agent




