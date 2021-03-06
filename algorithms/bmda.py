import numpy as np

class BMDA:
    class Solution:
        def __init__(self, bitstring, fitness) -> None:
            self.bitstring = bitstring
            self.fitness = fitness

    def __init__(self, fitness_function, validation_function, num_generations, population_size, parent_size, offspring_size) -> None:
        self.num_generations = num_generations
        self.fitness_function = fitness_function
        self.validation_function = validation_function
        self.offspring_size = offspring_size
        self.parent_size = parent_size
        self.population_size = population_size

    def generate_single_solution(self, item_probability_vector, nr_of_features) -> Solution:
        bitstring = []
        for i in range(nr_of_features):
            if np.random.rand() <= item_probability_vector[i] and self.validation_function(bitstring):
                bitstring.append(1)
            else:
                bitstring.append(0)
        
        return self.Solution(bitstring, self.fitness_function(bitstring))

    def generate_random_population(self, nr_of_items):
        uniform_dist = [0.5 for _ in range(nr_of_items)]
        return [self.generate_single_solution(uniform_dist, nr_of_items) for _ in range(self.population_size)]

    def calculate_distributions(self, parents, nr_of_features):
        univariate_freqs = [0 for _ in range(nr_of_features)]
        bivariate_freqs = [[np.zeros((2, 2)) for __ in range(nr_of_features)] for _ in range(nr_of_features)]

        for solution in parents:
            for index1 in range(len(solution.bitstring)):
                item1 = solution.bitstring[index1]
                # Univariate calculation
                if item1 == 1:
                    univariate_freqs[index1] += 1

                if index1 != len(solution.bitstring) - 1:
                    for index2 in range(index1+1, len(solution.bitstring)):
                        # Bivariate calculation
                        item2 = solution.bitstring[index2]
                        bivariate_freqs[index1][index2][item1][item2] += 1
                        bivariate_freqs[index2][index1][item2][item1] += 1


        for index in range(len(univariate_freqs)):
            univariate_freqs[index] /= self.parent_size

        for i in range(len(bivariate_freqs)):
            for j in range(i+1, len(bivariate_freqs)):
                bivariate_freqs[i][j][0][0] /= self.parent_size
                bivariate_freqs[i][j][0][1] /= self.parent_size
                bivariate_freqs[i][j][1][0] /= self.parent_size
                bivariate_freqs[i][j][1][1] /= self.parent_size

                bivariate_freqs[j][i][0][0] /= self.parent_size
                bivariate_freqs[j][i][0][1] /= self.parent_size
                bivariate_freqs[j][i][1][0] /= self.parent_size
                bivariate_freqs[j][i][1][1] /= self.parent_size

        return univariate_freqs, bivariate_freqs

    def find_dependencies(self, univariate_freqs, bivariate_freqs):
        dependencies = [[] for _ in univariate_freqs]
        for index1 in range(len(bivariate_freqs)-1):
            for index2 in range(index1+1, len(bivariate_freqs)):
                matrix = bivariate_freqs[index1][index2]
                independent_dist00 = self.parent_size * (1. - univariate_freqs[index1]) * (1. - univariate_freqs[index2])
                dependent_dist00 = self.parent_size * matrix[0][0]
                independent_dist01 = self.parent_size * (1. - univariate_freqs[index1]) * (univariate_freqs[index2])
                dependent_dist01 = self.parent_size * matrix[0][1]
                independent_dist10 = self.parent_size * (univariate_freqs[index1]) * (1. - univariate_freqs[index2])
                dependent_dist10 = self.parent_size * matrix[1][0]
                independent_dist11 = self.parent_size * (univariate_freqs[index1]) * (univariate_freqs[index2])
                dependent_dist11 = self.parent_size * matrix[1][1]
                chi_square = 0
                if (independent_dist00 != 0):
                    chi_square += ((dependent_dist00 - independent_dist00)**2) / independent_dist00
                if (independent_dist01 != 0):
                    chi_square += ((dependent_dist01 - independent_dist01)**2) / independent_dist01
                if (independent_dist10 != 0):
                    chi_square += ((dependent_dist10 - independent_dist10)**2) / independent_dist10
                if (independent_dist11 != 0):
                    chi_square += ((dependent_dist11 - independent_dist11)**2) / independent_dist11

                if (chi_square >= 3.84):
                    dependencies[index1].append((index2, chi_square))
            
        for dep in dependencies:
            dep.sort(key=lambda x: x[1], reverse=True)

        return dependencies

    def construct_dependency_graph(self, nr_of_features, dependencies):
        not_added_to_graph = [i for i in range(nr_of_features)]
        added_to_graph = []
        edges = []
        special_set = []
        for vertex in not_added_to_graph:
            if vertex in added_to_graph:
                continue

            special_set.append(vertex)
            added_to_graph.append(vertex)
            while True:
                # Find all dependencies of vertices already in graph and pick highest
                greatest_dep = None
                greatest_index = -1
                for index in range(nr_of_features):
                    if index in added_to_graph:
                        dep = dependencies[index]
                        if len(dep) != 0 and (greatest_dep is None or dep[0][1] >= greatest_dep[1]):
                            greatest_dep = dep[0]
                            greatest_index = index
                if greatest_dep is not None and greatest_dep[0] not in added_to_graph:
                    added_to_graph.append(greatest_dep[0])
                    edges.append((greatest_index, greatest_dep[0]))
                    dependencies[greatest_index].pop(0)
                else:
                    break

        return (added_to_graph, edges, special_set)

    def generate_new_individual(self, dependency_graph, univariate_freqs, bivariate_freqs):
        vertices, edges, special_set = dependency_graph
        individual = [0 for _ in vertices]
        K = vertices[:]
        for k in special_set:
            if np.random.rand() <= univariate_freqs[k]:
                individual[k] = 1
            else:
                individual[k] = 0
            K.remove(k)

        if (len(K) != 0):
            for edge in edges:
                if (edge[1] in K and edge[0] not in K):
                        in_k = edge[1]
                        not_in_k = edge[0]
                        if (individual[not_in_k] == 0):
                            is1 = bivariate_freqs[in_k][not_in_k][1][individual[not_in_k]] / (1. - univariate_freqs[not_in_k])
                        else:
                            is1 = bivariate_freqs[in_k][not_in_k][1][individual[not_in_k]] / univariate_freqs[not_in_k]
                            
                        if np.random.rand() <= is1:
                            individual[in_k] = 1
                        else:
                            individual[in_k] = 0

                        K.remove(in_k)

        return individual

    def parent_selection(self, population):
        population.sort(key=lambda x: x.fitness, reverse=True)
        return population[:self.parent_size]

    def generate_offspring(self, dependency_graph, univariate_freqs, bivariate_freqs):
        offspring = []
        for _ in range(self.offspring_size):
            while True:
                new_individual = self.generate_new_individual(dependency_graph, univariate_freqs, bivariate_freqs)
                individual_fitness = self.fitness_function(new_individual)
                if (self.validation_function(new_individual)):
                    offspring.append(self.Solution(new_individual, individual_fitness))
                    break

        return offspring

    def calculate(self, nr_of_features) -> int:
        population = self.generate_random_population(nr_of_features)
        best_results = []
        for generation in range(self.num_generations):
            parents = self.parent_selection(population)
            univariate_dist, bivariate_dist = self.calculate_distributions(parents, nr_of_features)
            dependencies = self.find_dependencies(univariate_dist, bivariate_dist)
            dependency_graph = self.construct_dependency_graph(nr_of_features, dependencies)
            offspring = self.generate_offspring(dependency_graph, univariate_dist, bivariate_dist)
            population += offspring
            population.sort(key=lambda x: x.fitness, reverse=True)
            population = population[:self.population_size]
            best_results.append(population[0])

        return best_results

