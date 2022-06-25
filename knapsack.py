from algorithms.bmda import BMDA
from algorithms.umda import UMDA
from matplotlib import pyplot as plt
import numpy as np
import time
import os

# Each item is represented with (weight, value, [(dependency_index1, bonus_points1), (dependency_index2, bonus_points2), ...])
ITEMS = []

NUM_GENERATIONS = 20
POPULATION_SIZE = 400
OFFSPRING_SIZE = 30
BAG_WEIGHT = 800
PARENT_SIZE = 30

def is_solution_valid(bitstring) -> bool:
        if calculate_weight(bitstring) > BAG_WEIGHT:
            return False

        return True

def calculate_weight(bitstring):
    weight = 0
    for index, element in enumerate(bitstring):
        if element == 1:
            weight += ITEMS[index][0]

    return weight

def fitness_function(items_bitstring) -> tuple:
    fitness = 0
    for index, element in enumerate(items_bitstring):
        if element == 1:
            fitness += ITEMS[index][1]
            for dependency in ITEMS[index][2]:
                if items_bitstring[dependency[0]] == 1:
                    fitness += dependency[1]

    return fitness

def plot_fitness(fitness_values):
    plt.figure()
    x = [i for i in range(NUM_GENERATIONS)]
    plt.xlabel("Generation")
    plt.ylabel("(Fitness, Weight)")
    plt.plot(fitness_values)
    plt.show()

def parse_items_file(file_path: str) -> list:
    items = []
    total_weight = 0
    with open(file_path) as file:
        for line in file:
            strings = line.split(',')
            weight = int(strings[0])
            total_weight += weight
            value = int(strings[1].strip(" "))
            dependencies = []
            for index in range(2, len(strings)):
                string = strings[index].strip(" '[]\n")
                strings[index] = string
                if len(string) != 0:
                    dep, bonus = string.split(" ")
                    dependencies.append((int(dep), int(bonus)))
            items.append((weight, value, dependencies))
        file.close()

    global BAG_WEIGHT
    BAG_WEIGHT = total_weight / 2.
    return items

def main():
    global ITEMS
    current_dir = os.path.dirname(__file__)
    ITEMS = parse_items_file(current_dir + "\\knapsack_data\\dataset2.txt")

    umda = UMDA(
        fitness_function,
        is_solution_valid,
        NUM_GENERATIONS,
        POPULATION_SIZE,
        PARENT_SIZE,
        OFFSPRING_SIZE,
    )

    bmda = BMDA(
        fitness_function,
        is_solution_valid,
        NUM_GENERATIONS,
        POPULATION_SIZE,
        PARENT_SIZE,
        OFFSPRING_SIZE,
    )

    # Multivariate
    # boa = BOA(

    # )

    start = time.perf_counter()
    fitness_values_umda = umda.calculate(len(ITEMS))
    end_umda = time.perf_counter() - start

    start = time.perf_counter()
    fitness_values_bmda = bmda.calculate(len(ITEMS))
    end_bmda = time.perf_counter() - start

    # start = time.perf_counter()
    # fitness_values_boa = boa.calculate(ITEMS)
    # end_boa = time.perf_counter()

    plot_fitness([(solution.fitness, calculate_weight(solution.bitstring)) for solution in fitness_values_umda])
    plot_fitness([(solution.fitness, calculate_weight(solution.bitstring)) for solution in fitness_values_bmda])
    # plot_fitness(fitness_values_boa)
    print("UMDA best:", fitness_values_umda[-1].fitness, "Elapsed time:", end_umda)
    print("BMDA best:", fitness_values_bmda[-1].fitness, "Elapsed time:", end_bmda)
    print("Maximum bag weight:", BAG_WEIGHT)
    # print("BOA best:", fitness_values_boa[-1])
    

if __name__ == "__main__":
    main()