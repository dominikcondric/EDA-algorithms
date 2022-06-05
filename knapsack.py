from bmda import BMDA
from matplotlib import pyplot as plt
import numpy as np

# Each item is represented with (weight, value, [dependency_index1, dependency_index2, ...])
ITEMS = []

NUM_GENERATIONS = 50
POPULATION_SIZE = 100
OFFSPRING_SIZE = 20
BAG_WEIGHT = 0
PARENT_SIZE = 20
ITEMS_SIZE = 70

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
    sum_weight = 0
    for i in range(ITEMS_SIZE):
        weight = np.random.randint(20)
        sum_weight += weight
        ITEMS.append((weight, np.random.randint(30), [np.random.randint(ITEMS_SIZE) for _ in range(np.random.randint(ITEMS_SIZE // 2))]))
    return sum_weight

def main():
    sum_weights = generate_random_items()
    BAG_WEIGHT = 250

    # Univariate
    # umda = UMDA(

    # )

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

    # fitness_values_umda = umda.calculate(ITEMS)
    fitness_values_bmda = bmda.calculate(ITEMS)
    # fitness_values_boa = boa.calculate(ITEMS)

    # plot_fitness(fitness_values_umda)
    plot_fitness(fitness_values_bmda)
    # plot_fitness(fitness_values_boa)
    # print("UMDA best:", fitness_values_umda[-1])
    print("Maximum bag weight:", BAG_WEIGHT)
    print("BMDA best:", fitness_values_bmda[-1])
    # print("BOA best:", fitness_values_boa[-1])
    

if __name__ == "__main__":
    main()