import numpy as np

class UMDA:
    class Solution:
        def __init__(self, bitstring, weight, fitness) -> None:
            self.bitstring = bitstring
            self.fitness = fitness
            self.weight = weight

    def __init__(self, fitness_function, num_generations, population_size, parent_size, offspring_size, bag_capacity) -> None:
        self.num_generations = num_generations
        self.fitness_function = fitness_function
        self.offspring_size = offspring_size
        self.parent_size = parent_size
        self.population_size = population_size
        self.bag_capacity = bag_capacity

    def is_solution_valid(self, solution) -> bool:
        if solution[0] > self.bag_capacity:
            return False

        return True

    def generate_single_solution(self, item_probability_vector, items) -> Solution:
        while (True):
            bitstring = []
            for i in range(len(items)):
                if np.random.rand() <= item_probability_vector[i]:
                    bitstring.append(1)
                else:
                    bitstring.append(0)
            
            fitness = self.fitness_function(bitstring)
            if (self.is_solution_valid(fitness)):
                return self.Solution(bitstring, *fitness)

    def generate_random_population(self, items):
        return [self.generate_single_solution([0.5 for _ in items], items) for _ in range(self.population_size)]

    def calculate_distribution(self, parents, items):
        univariate_freqs = [0 for _ in items]

        for solution in parents:
            for index1 in range(len(solution.bitstring)):
                item1 = solution.bitstring[index1]
                if item1 == 1:
                    univariate_freqs[index1] += 1

        for index in range(len(univariate_freqs)):
            univariate_freqs[index] /= self.parent_size
            

        return univariate_freqs


    def generate_new_individual(self, univariate_freqs):
        individual = [0 for _ in univariate_freqs]

        counter = 0
        for chance in univariate_freqs:
            if np.random.rand() <= chance:
                individual[counter] = 1    
            else:
                individual[counter] = 0                        
            counter += 1

        return individual

    def parent_selection(self, population):
        population.sort(key=lambda x: x.fitness, reverse=True)
        return population[:self.parent_size]

    def generate_offspring(self, univariate_freqs):
        offspring = []
        for _ in range(self.offspring_size):
            while True:
                new_individual = self.generate_new_individual(univariate_freqs)
                individual_fitness = self.fitness_function(new_individual)
                if (self.is_solution_valid(individual_fitness)):
                    offspring.append(self.Solution(new_individual, *individual_fitness))
                    break

        return offspring

    def calculate(self, items):
        population = self.generate_random_population(items)
        best_results = []
        for generation in range(self.num_generations):
            print(f"Generation: {generation+1}")
            parents = self.parent_selection(population)
            univariate_dist = self.calculate_distribution(parents, items)
            offspring = self.generate_offspring(univariate_dist)
            population += offspring
            population.sort(key=lambda x: x.fitness, reverse=True)
            population = population[:self.population_size]
            best_results.append((population[0].fitness, population[0].weight))

        return best_results