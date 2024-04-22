import numpy as np

class ABC:
    def __init__(self, problem, population_size, iterations):
        self.problem = problem
        self.population_size = population_size
        self.iterations = iterations

    def initialization_phase(self):
        population = [self.problem.random_solution() for _ in range(self.population_size)]
        return population

    def employed_bees_phase(self, population):
        for i, solution in enumerate(population):
            neighbor_index = np.random.choice(range(self.population_size))
            parameter_index = np.random.randint(self.problem.dimension)
            phi = np.random.uniform(-1, 1)
            neighbor_solution = solution + phi * (solution - population[neighbor_index])
            neighbor_solution = np.clip(neighbor_solution, self.problem.lower_bound, self.problem.upper_bound)
            neighbor_fitness = self.problem.objective_function(neighbor_solution)

            if neighbor_fitness < self.problem.objective_function(solution):
                population[i] = neighbor_solution

        return population

    def onlooker_bees_phase(self, population):
        fitness_values = [self.problem.objective_function(solution) for solution in population]
        probabilities = [fit / sum(fitness_values) for fit in fitness_values]

        onlooker_population = []
        for _ in range(self.population_size):
            selected_index = np.random.choice(range(self.population_size), p=probabilities)
            onlooker_population.append(population[selected_index])

        return onlooker_population

    def scout_bees_phase(self, population):
        for i, solution in enumerate(population):
            if np.random.rand() < 0.1:  # Probability threshold for abandoning the solution
                population[i] = self.problem.random_solution()

        return population

    def optimize(self):
        population = self.initialization_phase()
        best_solution = None
        best_fitness = np.inf

        for _ in range(self.iterations):
            employed_population = self.employed_bees_phase(population)
            onlooker_population = self.onlooker_bees_phase(employed_population)
            population = self.scout_bees_phase(onlooker_population)

            current_best_fitness = min([self.problem.objective_function(sol) for sol in population])
            if current_best_fitness < best_fitness:
                best_solution = population[np.argmin([self.problem.objective_function(sol) for sol in population])]
                best_fitness = current_best_fitness

        return best_solution, best_fitness


