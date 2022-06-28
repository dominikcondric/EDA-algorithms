from audioop import add
from bayes_optim import Solution
import numpy as np
from algorithms.boa_utils.bayesian_network import BayesianNetwork
import time

class BOA:
    class Solution:
        def __init__(self, bitstring, fitness) -> None:
            self.bitstring = bitstring
            self.fitness = fitness

    def __init__(self, fitness_function, is_solution_valid, num_generations, population_size, parent_size, offspring_size, setup, log=False) -> None:
        self.num_generations = num_generations
        self.fitness_function = fitness_function
        self.is_solution_valid = is_solution_valid
        self.offspring_size = offspring_size
        self.parent_size = parent_size
        self.population_size = population_size
        self.log = log

        self.setup = setup

        self.solution = [0, 0]
        self.solution_num = 0

    def generate_single_solution(self, item_probability_vector, nr_of_items) -> Solution:
        while (True):
            bitstring = []
            for i in range(nr_of_items):
                if np.random.rand() <= item_probability_vector[i]:
                    bitstring.append(1)
                else:
                    bitstring.append(0)
            
            fitness = self.fitness_function(bitstring)
            if (self.is_solution_valid(bitstring)):
                return self.Solution(bitstring, fitness)

    def generate_random_population(self, nr_of_items):
        return [self.generate_single_solution([0.5 for _ in range(nr_of_items)], nr_of_items) for _ in range(self.population_size)]

    def parent_selection(self, population):
        population.sort(key=lambda x: x.fitness, reverse=True)
        return population[:self.parent_size]

    def calculate(self, nr_of_items) -> int:
        population = self.generate_random_population(nr_of_items)
        best_results = []

        for generation in range(self.num_generations):
            if self.log: print(f"Generation: {generation+1}")
            parents = self.parent_selection(population) # sort population

            bn = BayesianNetwork(parents, nr_of_items)
            if self.log: print('Estimating edges...', end=' ')
            time0 = time.process_time()

            bn.estimate_edges()
            time1 = time.process_time()
            if self.log: print('time: ' + (str) (time1 - time0) + ' sec')
            if self.log: print('Estimating params...', end=' ')

            bn.estimate_params()
            time2 = time.process_time()
            if self.log: print('time: ' + (str) (time2 - time1) + ' sec')
            if self.log: print('Calculating samples...', end=' ')

            offspring = self.setup.get_elites(population)
            elite_size = int (self.setup.elite_rate * len(population))
            for i in range(self.population_size - elite_size):

                while True:
                    bitstring = bn.sampling()
                    Individual_fitness = self.fitness_function(bitstring)

                    if (self.is_solution_valid(bitstring)):
                        offspring.append(self.Solution(bitstring, Individual_fitness))
                        break

            time3 = time.process_time()
            if self.log: print ('time: ' + (str) (time3 - time2) + ' sec')
            population += offspring

            population.sort(key=lambda x: x.fitness, reverse=True)
            population = population[:self.population_size]
            if self.log: print('BOA best result: fitness-' +  (str) (population[0].fitness))
            best_results.append(population[0])

            # checking if solution is changed, if not then break
            if (population[0].fitness != self.solution[0]):
                self.solution[0] = population[0].fitness
                self.solution_num = 0
            else:
                self.solution_num += 1

            if self.solution_num >= 5:
                n = len(best_results)
                for i in range(n, self.num_generations):
                    best_results.append(population[0])
                if self.log: print('BOA solution hasn\'t changed. Breaking...')

                break

        return best_results

