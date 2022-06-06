from bmda import BMDA
from umda import UMDA
from matplotlib import pyplot as plt
import numpy as np
import time

# Each item is represented with (weight, value, [dependency_index1, dependency_index2, ...])
ITEMS = []

NUM_GENERATIONS = 5
POPULATION_SIZE = 200
OFFSPRING_SIZE = 50
BAG_WEIGHT = 1000
PARENT_SIZE = 50
ITEMS_SIZE = 150

def fitness_function(items_bitstring) -> tuple:
    fitness = 0
    weight = 0
    for index, element in enumerate(items_bitstring):
        if element == 1:
            weight += ITEMS[index][0]
            fitness += ITEMS[index][1]
            for dependency in ITEMS[index][2]:
                if items_bitstring[dependency] == 1:
                    fitness += 1

    return (weight, fitness)

def plot_fitness(fitness_values):
    plt.figure()
    x = [i for i in range(NUM_GENERATIONS)]
    plt.xlabel("Generation")
    plt.ylabel("(weight, fitness)")
    plt.plot(fitness_values, label=("fitness", "weight"))
    plt.legend()
    plt.show()

def generate_random_items():
    for i in range(ITEMS_SIZE):
        weight = np.random.randint(20)
        ITEMS.append((weight, np.random.randint(30), [np.random.randint(ITEMS_SIZE) for _ in range(np.random.randint(ITEMS_SIZE // 2))]))

def main():
    generate_random_items()

    umda = UMDA(
        fitness_function,
        NUM_GENERATIONS,
        POPULATION_SIZE,
        PARENT_SIZE,
        OFFSPRING_SIZE,
        BAG_WEIGHT
    )

    bmda = BMDA(
        fitness_function,
        NUM_GENERATIONS,
        POPULATION_SIZE,
        PARENT_SIZE,
        OFFSPRING_SIZE,
        BAG_WEIGHT
    )

    # Multivariate
    # boa = BOA(

    # )

    start = time.perf_counter()
    fitness_values_umda = umda.calculate(ITEMS)
    end_umda = time.perf_counter() - start

    start = time.perf_counter()
    fitness_values_bmda = bmda.calculate(ITEMS)
    end_bmda = time.perf_counter() - start

    # start = time.perf_counter()
    # fitness_values_boa = boa.calculate(ITEMS)
    # end_boa = time.perf_counter()

    plot_fitness(fitness_values_umda)
    plot_fitness(fitness_values_bmda)
    # plot_fitness(fitness_values_boa)
    print("UMDA best:", fitness_values_umda[-1], "Elapsed time:", end_umda)
    print("Maximum bag weight:", BAG_WEIGHT)
    print("BMDA best:", fitness_values_bmda[-1], "Elapsed time:", end_bmda)
    # print("BOA best:", fitness_values_boa[-1])
    

if __name__ == "__main__":
    main()