# Author: Connor Anderson
# Last Modified Date: November 30, 2017
from abc import ABC, abstractmethod
from ordered_set import OrderedSet
import random


class GeneticAlgorithm(ABC):
    """Abstract class that allows for customized implementation of general Genetic Algorithm (GA).
    """
    def __init__(self, selection_function, fitness_function, crossover_function,
                 mutation_function, population, num_generations=500):
        self._selection_function = selection_function
        self._fitness_function = fitness_function
        self._crossover_function = crossover_function
        self._mutation_function = mutation_function
        self._population = population
        self._num_generations = num_generations

    def run(self):
        current_population = self._population
        for generation in range(self._num_generations):
            current_population = self.create_new_generation(current_population)
        return self._selection_function(current_population, self._fitness_function).optimum_value

    @abstractmethod
    def create_new_generation(self, current_population):
        pass


class PopulationInitializer(ABC):
    """Abstract class that allows for customized implementation of general Population Initializer for GA's.
    """
    def __init__(self, alphabet):
        self._alphabet = alphabet

    def create_population(self, population_size, chromosome_length=None):
        population = []
        for organism in range(population_size):
            population.append(self.create_organism(chromosome_length))
        return tuple(population)

    @abstractmethod
    def create_organism(self, chromosome_length):
        pass


class PermutationPopulationInitializer(PopulationInitializer):
    """Implementation class that allows for creation of a permuted population for use in GA's.
    """
    def create_organism(self, chromosome_length=None):
        if chromosome_length is None:
            chromosome_length = len(self._alphabet)
        new_organism = random.sample(list(self._alphabet), chromosome_length)
        return tuple(new_organism)


# TODO: Separate use of fitness function from selection function
class SelectionFunction(ABC):
    """Abstract class that allows for customized implementation of general Selection Function for GA's.
    """
    def __init__(self, population, fitness_function):
        self._population = population
        self._fitness_function = fitness_function
        fitness_values = {}
        for organism in population:
            if organism in fitness_values:
                continue
            fitness_values[organism] = self._fitness_function(organism)
        self._sorted_organisms = sorted(fitness_values, key=fitness_values.get)

    def __call__(self):
        return self.select()

    @property
    def optimum_value(self):
        return self._sorted_organisms[0]

    @abstractmethod
    def select(self):
        pass


class TopTenPercentSelectionFunction(SelectionFunction):
    """Implementation class that selects the top 10 percent of a population for subsequent breeding.
    """
    def select(self):
        population_size = len(self._population)
        ten_percent_index = round(population_size * 0.1)
        return self._sorted_organisms[:ten_percent_index]


class FitnessFunction(ABC):
    """Abstract class that allows for customized implementation of general Fitness Function for use in Selection Function.
    """
    def __call__(self, organism):
        return self.evaluate(organism)

    @abstractmethod
    def evaluate(self, organism):
        pass


class CrossoverFunction(ABC):
    """Abstract class that allows for customized implementation of general Crossover Function for GA's.
    """
    def __init__(self, parent_pool, num_offspring):
        self._parent_pool = parent_pool
        self._num_offspring = num_offspring

    def __call__(self):
        new_generation = []
        for parent in self._parent_pool:
            new_generation.append(parent)
        return tuple(new_generation + self.breed())

    def breed(self):
        offspring = []
        for i in range(self._num_offspring):
            parents = random.sample(self._parent_pool, 2)
            offspring.append(self.crossover(parents[0], parents[1]))
        return offspring

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass


class OrderCrossoverFunction(CrossoverFunction):
    """Implementation class that allows for crossover in organisms whose chromosomes are order-dependent.
    """
    def __init__(self, parent_pool, num_offspring, crossover_prob=25):
        if crossover_prob <= 0 or crossover_prob > 100:
            raise ValueError("Crossover probability must be within range of 0 to 100")

        self._crossover_prob = crossover_prob
        super(OrderCrossoverFunction, self).__init__(parent_pool, num_offspring)

    def crossover(self, parent1, parent2):
        parent1_length = len(parent1)
        parent2_length = len(parent2)
        if parent1_length != parent2_length:
            raise ValueError("Parents must be of same length")

        offspring = []

        allele_ordered_set = OrderedSet()
        for allele in parent1:
            if RandomUtils.randboolweighted(self._crossover_prob):
                allele_ordered_set.append(allele)

        crossover_set_position = 0
        for allele in parent2:
            if allele in allele_ordered_set:
                offspring.append(allele_ordered_set[crossover_set_position])
                crossover_set_position += 1
            else:
                offspring.append(allele)
        return tuple(offspring)


class MutationFunction(ABC):
    """Abstract class that allows for customized implementation of general Mutation Function for GA's.
    """
    def __init__(self, population):
        self._population = list(population)

    def __call__(self):
        for index, organism in enumerate(self._population):
            if RandomUtils.randboolweighted(5):
                mutated_organism = self.mutate(organism)
                self._population[index] = mutated_organism
        return tuple(self._population)

    @abstractmethod
    def mutate(self, organism):
        pass


class PointMutationFunction(MutationFunction):
    """Implementation class that allows for point to point mutation of organisms in a GA.
    """
    def mutate(self, organism):
        organism = list(organism)
        locus1 = random.randint(0, len(organism) - 1)
        locus2 = locus1

        while locus1 == locus2:
            locus2 = random.randint(0, len(organism) - 1)
            organism[locus1], organism[locus2] = organism[locus2], organism[locus1]
        return tuple(organism)


class ShuffleMutationFunction(MutationFunction):
    """Implementation class that allows for a mutation based on shifting a subset of an organism's chromosome in a GA.
    """
    def __init__(self, population, shuffle_length=None):
        if shuffle_length is None:
            if len(population) > 10:
                shuffle_length = 10
            else:
                shuffle_length = len(population) - 1

        if shuffle_length <= 0 or shuffle_length >= len(population):
            raise ValueError("Length of shuffle must be greater than 0 and less than the size of the population")

        self._shuffle_length = shuffle_length
        super(ShuffleMutationFunction, self).__init__(population)

    def mutate(self, organism):
        start_locus = random.randint(0, len(organism) - 1)
        insertion_locus = start_locus
        while start_locus == insertion_locus:
            insertion_locus = random.randint(0, len(organism) - 1)

        mutation_length = random.randint(1, self._shuffle_length)
        mutation_subset = organism[start_locus:start_locus + mutation_length]

        mutated_organism = organism[:start_locus] + organism[start_locus + mutation_length:]

        while start_locus == insertion_locus:
            insertion_locus = random.randint(0, len(mutated_organism) - 1)
        return tuple(mutated_organism[:insertion_locus] + mutation_subset + mutated_organism[insertion_locus:])


class RandomUtils:
    """Utility class to handle miscellaneous functions.
    """
    @staticmethod
    def randboolweighted(percentage=50):
        if percentage <= 0 or percentage > 100:
            raise ValueError("Percentage must be within range of 0 to 100")

        return random.randint(0, 99) < percentage
