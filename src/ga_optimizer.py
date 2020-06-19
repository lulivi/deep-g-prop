"""Multilayer peceptron optimization via genetic algorithms."""
import random
import time

from typing import Callable, List, Tuple

import numpy as np

from deap import base, creator, tools
from keras import Sequential
from sklearn.metrics import balanced_accuracy_score, fbeta_score

from src.common import SEED
from src.dgp_logger import DGPLOGGER
from src.types import Proben1Partition, Proben1Split
from src.utils import print_table

ModelStructure = List[Tuple[tuple, np.dtype]]
np.set_printoptions(precision=5, floatmode="fixed")


class MLPIndividual:
    """Basic structure of a simple MLP weights."""

    def __init__(self) -> None:
        """Create an empty weights list."""
        self._weights: List[np.array] = []
        self._bias: List[np.array] = []

    def __len__(self) -> int:
        """Calculate the number of layers the object has."""
        return len(self._bias)

    def add_layer(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """Append a new layer.

        :param weights: layer weights.
        :param bias: layer bias.

        """
        self._weights.append(weights)
        self._bias.append(bias)

    @property
    def weights(self) -> np.ndarray:
        """Obtain the individual weights.

        :returns: the weights :class:`np.ndarray`.

        """
        return self._weights

    @property
    def bias(self) -> np.ndarray:
        """Obtain the individual bias.

        :returns: the bias :class:`np.ndarray`.

        """
        return self._bias

    def get_all(self) -> List[np.array]:
        """Return the hole structure.

        :returns: the layer list.

        """
        return [
            layer
            for layer_tuple in zip(self._weights, self._bias)
            for layer in layer_tuple
        ]

    def can_mate(self, other: "MLPIndividual") -> bool:
        """Check if the current individual can mate with ``other``.

        :param other: other individual.
        :returns: ``True`` if both individuals can mate.

        """
        if len(self) != len(other):
            return False

        for layer, other_layer in zip(self.get_all(), other.get_all()):
            if layer.shape != other_layer.shape:
                return False

        return True

    def __str__(self) -> str:
        """Serialize as a string the object."""
        str_representation = f"{self.__class__.__name__}:"

        for index, (weights, bias) in enumerate(
            zip(self._weights, self._bias)
        ):
            str_representation += (
                f"\nLayer {index}:"
                f"\n-- Weights --\n{str(weights)}"
                f"\n-- Bias --\n{str(bias)}"
            )

        return str_representation

    def __repr__(self) -> str:
        """Serialize as a string the object."""
        return str(self)


###############################################################################
#
# Toolbox functions
#
###############################################################################


def individual_initializer(
    individual_class, weight_structure, kernel_initializer=np.random.uniform,
):
    """Initialize an individual.

    :param individual_class: individual class.
    :param weight_structure: the internal array of weights the individual has.
    :param kernel_initializer: weigths layers initializer.

    """
    individual = individual_class()

    for weights, bias in zip(weight_structure[::2], weight_structure[1::2]):
        individual.add_layer(
            kernel_initializer(low=-1.0, high=1.0, size=weights[0]),
            np.zeros(shape=bias[0], dtype=bias[1]),
        )

    return individual


def individual_evaluator(
    individual: MLPIndividual,
    model: Sequential,
    trn: Proben1Split,
    tst: Proben1Split,
    **kwargs,
):
    """Evaluate an individual.

    :param individual: current individual to evaluate.
    :param model: :class:`keras.Sequential` model to evaluate the individual
        with.
    :param trn: training data and labels.
    :param tst: validation data and labels.
    :param **kwargs: See below.

    :Keyword Arguments:
        - fit_train: whether to fit the train data with some forward pases
            before predicting.
        - average: average approach for :func:`fbeta_score` scorer.
    :returns: a tuple with the scores.

    """
    model.set_weights(individual.get_all())

    if kwargs.pop("fit_train", False):
        model.fit(
            trn.X, trn.y_cat, epochs=20, batch_size=8, verbose=0,
        )

    predicted_y = model.predict_classes(tst.X)
    f2_score = fbeta_score(
        tst.y,
        predicted_y,
        beta=2,
        average=kwargs.pop(
            "average", "micro" if trn.y_cat.shape[1] > 2 else "binary"
        ),
    )
    bal_accuracy_score = balanced_accuracy_score(tst.y, predicted_y)
    DGPLOGGER.debug(
        f"    Obtained scores: f2-score={f2_score:.5f}, "
        f"balanced acc score={bal_accuracy_score:.5f}"
    )

    return (f2_score, bal_accuracy_score)


def crossover_operator(ind1: MLPIndividual, ind2: MLPIndividual):
    """Apply crossover betweent two individuals.

    This method will swap a random neuron from a random layer. The neuron
    associated bias and weights are swapped.

    :param ind1: the first individual.
    :param ind2: the second individual.

    """
    layer = random.randint(0, len(ind1) - 1)
    neuron = random.randint(0, len(ind1.bias[layer]) - 1)

    (ind1.weights[layer][:, neuron], ind2.weights[layer][:, neuron],) = (
        ind2.weights[layer][:, neuron].copy(),
        ind1.weights[layer][:, neuron].copy(),
    )
    (ind1.bias[layer][neuron], ind2.bias[layer][neuron]) = (
        ind2.bias[layer][neuron],
        ind1.bias[layer][neuron],
    )


def bias_mutator(individual: MLPIndividual, gen_prob: float) -> int:
    """Mutate some individual bias genes.

    For each bias layer, obtain a random (between 0.0 and 1.0)
    :class:`np.ndarray` with the same shape of the bias (in this case, a 1D
    numpy array) and mutate the genes that satisfy the ``gen_prob`` probability
    to mutate.

    :param individual: individual to mutate.
    :param gen_prob: probability of a gen to mutate.
    :returns: number of genes mutated.

    """
    mutated_genes = 0

    for layer_idx in range(len(individual)):
        mask = np.random.rand(*individual.bias[layer_idx].shape) < gen_prob
        mutated_genes += np.count_nonzero(mask)
        individual.bias[layer_idx][mask] += random.uniform(-0.5, 0.5)

    return mutated_genes


def weights_mutator(individual: MLPIndividual, gen_prob: float) -> int:
    """Mutate some individual weights genes.

    :param individual: individual to mutate.
    :param gen_prob: probability of a gen to mutate.
    :returns: number of genes mutated.

    """
    mutated_genes = 0

    for layer_idx in range(len(individual)):
        mask = np.random.rand(*individual.weights[layer_idx].shape) < gen_prob
        mutated_genes += np.count_nonzero(mask)
        individual.weights[layer_idx][mask] += random.uniform(-0.5, 0.5)

    return mutated_genes


# pylint: disable=no-member
def configure_toolbox(
    model: Sequential,
    dataset: Proben1Partition,
    mut_bias_prob: float,
    mut_weights_prob: float,
    fit_train: bool,
):
    """Register all neccesary objects and functions.

    :param model: base model, from whom obtain the initial individual shape.
    :param dataset: data to work with.
    :param mut_bias_prob: probability to mutate the individual bias genes.
    :param mut_weights_prob: probability to mutate the individual weights
        genes.
    :param fit_train: whether to fit the training data in each evaluation.
    :returns: the toolbox with the registered functions.

    """
    # --------------------------------
    # Individual registration
    # --------------------------------
    DGPLOGGER.debug("-- Register necessary functions and elements")
    DGPLOGGER.debug("Register the fitness measure...")
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.7))

    DGPLOGGER.debug("Register the individual...")
    creator.create("Individual", MLPIndividual, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    DGPLOGGER.debug("Register the individual initializer...")
    toolbox.register(
        "individual",
        individual_initializer,
        creator.Individual,
        weight_structure=[
            (layer.shape, layer.dtype) for layer in model.get_weights()
        ],
    )

    # define the population to be a list of individuals
    DGPLOGGER.debug("Register the population initializer...")
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --------------------------------
    # Operator registration
    # --------------------------------
    DGPLOGGER.debug("Register the evaluator function...")
    toolbox.register(
        "evaluate",
        individual_evaluator,
        model=model,
        trn=dataset.trn,
        tst=dataset.val,
        average="micro" if dataset.nout > 2 else "binary",
        fit_train=fit_train,
    )

    DGPLOGGER.debug("Register the crossover operator...")
    toolbox.register("crossover", crossover_operator)

    DGPLOGGER.debug("Register the weights mutate operator...")
    toolbox.register(
        "mutate_weights", weights_mutator, gen_prob=mut_weights_prob
    )

    DGPLOGGER.debug("Register the bias mutate operator...")
    toolbox.register("mutate_bias", bias_mutator, gen_prob=mut_bias_prob)

    DGPLOGGER.debug("Register the selector function...")
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


###############################################################################
#
# Util functions
#
###############################################################################


def evaluate_population(population: list, evaluate_fn: Callable) -> None:
    """Evaluate the population.

    :param population: list of individuals.
    :param evaluate_fn: function to evaluate the population.

    """
    fitnesses = map(evaluate_fn, population)

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit


def apply_crossover(
    population: list, cx_prob: float, crossover_fn: Callable
) -> int:
    """Crossover the population by 2 with a probability.

    :param population: list of individuals.
    :param cx_prob: mate probability.
    :param crossover_fn: function to mate two individuals.
    :returns: number of individual pairs mated.

    """
    crossed_individuals = 0

    for child1, child2 in zip(population[::2], population[1::2]):
        if random.random() < cx_prob and child1.can_mate(child2):
            crossed_individuals += 1
            crossover_fn(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    return crossed_individuals


def apply_mutation(
    population: list,
    mut_bias_prob: float,
    mut_weights_prob: float,
    mut_bias_fn: Callable,
    mut_weights_fn: Callable,
) -> int:
    """Mutate the population with probabilities.

    :param population: list of individuals.
    :param mut_bias_prob: probability to mutate the individual bias genes.
    :param mut_weights_prob: probability to mutate the individual weights
        genes.
    :param mut_bias_fn: function to mute the bias genes.
    :param mut_weights_fn: function to mute the weights genes.
    :returns: the number of individuals mutated.

    """
    mutated_individuals = 0

    for index, mutant in enumerate(population):
        mutated_bias_genes = mutated_weights_genes = 0

        if random.random() < mut_bias_prob:
            mutated_bias_genes = mut_bias_fn(mutant)
            del mutant.fitness.values

        if random.random() < mut_weights_prob:
            mutated_weights_genes = mut_weights_fn(mutant)
            del mutant.fitness.values

        if mutated_bias_genes or mutated_weights_genes:
            mutated_individuals += 1
            DGPLOGGER.debug(
                f"    For individual {index}: mutated "
                f"{mutated_bias_genes} bias genes and "
                f"{mutated_weights_genes} weights genes."
            )

    return mutated_individuals


def finished_generation_summary(current_generation: int, population: list):
    """Print a summary of the current generation.

    :param current_generation: index of the generation.
    :param population: list of individuals of the current generation.

    """
    fits = np.array([ind.fitness.values for ind in population])
    table = [
        ["Statistic", "F2 score", "Balanced acc score"],
        ["Max", fits[:, 0].max(), fits[:, 1].max()],
        ["Avg", fits[:, 0].mean(), fits[:, 1].mean()],
        ["Min", fits[:, 0].min(), fits[:, 1].min()],
        ["Std", fits[:, 0].std(), fits[:, 1].std()],
    ]
    DGPLOGGER.info(f"    Sumary of generation {current_generation}:")
    print_table(table, DGPLOGGER.info, floatfmt=(None, ".5f", ".5f"))


def finished_algorithm_summary(
    initial_population, final_population, best_individual_selector
):
    """Print a summary of the hole process.

    :param initial_population: list of initial individuals.
    :param final_population: list of final individuals.
    :param best_individual_selector: function to select the best individual.

    """
    initial_pop_table = []
    final_pop_table = []
    table_common_attributes = {
        "headers": ["Index", "F2 score", "Balanced acc score"],
        "print_fn": DGPLOGGER.debug,
        "floatfmt": ("i", ".5f", ".5f"),
    }

    for index, individual in enumerate(
        best_individual_selector(initial_population, len(initial_population))
    ):
        initial_pop_table.append(
            [
                str(index),
                str(individual.fitness.values[0]),
                str(individual.fitness.values[1]),
            ]
        )

    DGPLOGGER.debug("Initial population fitness values:")
    print_table(initial_pop_table, **table_common_attributes)

    for index, individual in enumerate(
        best_individual_selector(final_population, len(final_population))
    ):
        final_pop_table.append(
            [
                str(index),
                str(individual.fitness.values[0]),
                str(individual.fitness.values[1]),
            ]
        )

    DGPLOGGER.debug("Final population fitness values:")
    print_table(final_pop_table, **table_common_attributes)


def test_individual(
    individual, model, dataset, fit_train,
) -> Tuple[float, float]:
    """Test an individual with the validation and test data.

    :param individual: current individual to evaluate.
    :param model: base model, from whom obtain the initial individual shape.
    :param dataset: data to work with.
    :param fit_train: whether to fit the training data in each evaluation.

    """
    DGPLOGGER.info("Best individual weights")
    DGPLOGGER.info(str(individual))
    DGPLOGGER.info(
        "Predicting the validation and test data with the best individual."
    )
    individual.get_all()
    val_scores = individual_evaluator(
        individual,
        model,
        dataset.trn,
        dataset.val,
        fit_train=fit_train,
        average="micro" if dataset.nout > 2 else "binary",
    )
    tst_scores = individual_evaluator(
        individual,
        model,
        dataset.trn,
        dataset.tst,
        fit_train=fit_train,
        average="micro" if dataset.nout > 2 else "binary",
    )
    individual_table = [
        ["Validation", str(val_scores[0]), str(val_scores[1])],
        ["Test", str(tst_scores[0]), str(tst_scores[1])],
    ]
    print_table(
        individual_table,
        print_fn=DGPLOGGER.info,
        floatfmt=(None, ".5f", ".5f"),
        headers=["Partition", "F2 score", "Balanced acc score"],
    )
    return tst_scores


###############################################################################
#
# Genetic Algofithm
#
###############################################################################


# pylint: disable=no-member,too-many-arguments,too-many-locals
def genetic_algorithm(
    model: Sequential,
    dataset: Proben1Partition,
    init_population_size: int = 100,
    max_generations: int = 1000,
    mut_bias_prob: float = 0.2,
    mut_weights_prob: float = 0.4,
    cx_prob: float = 0.5,
    fit_train: bool = False,
) -> Tuple[MLPIndividual, MLPIndividual]:
    """Perform optimization with a genetic algorithm.

    :param model: base model, from whom obtain the initial individual shape.
    :param dataset: data to work with.
    :param init_population_size: initial population size.
    :param max_generations: maximun number of generations to run.
    :param mut_bias_prob: probability to mutate the individual bias genes.
    :param mut_weights_prob: probability to mutate the individual weights
        genes.
    :param cx_prob: probability to mate two individuals.
    :param fit_train: whether to fit the training data in each evaluation.
    :returns: the toolbox with the registered functions.

    """
    np.random.seed(SEED)
    random.seed(SEED)

    # Configure dataset related variables
    toolbox = configure_toolbox(
        model, dataset, mut_bias_prob, mut_weights_prob, fit_train,
    )

    # --------------------------------
    # Algorithm start
    # --------------------------------
    DGPLOGGER.title(msg="Start the algorithm")
    time_start = time.perf_counter()
    population = toolbox.population(n=init_population_size)

    DGPLOGGER.debug("Evaluate the initial population")
    evaluate_population(population, toolbox.evaluate)

    population.sort(key=lambda ind: ind.fitness.values, reverse=True)
    test_individual(population[0], model, dataset, fit_train)

    initial_population = population[:]
    DGPLOGGER.debug(f"    -- Evaluated {len(population)} individuals")

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in population]

    # Variable keeping track of the number of generations
    current_generation = 0

    try:
        # Begin the evolution
        while max(fits) < 1 and current_generation < max_generations:
            # A new generation
            current_generation = current_generation + 1
            DGPLOGGER.info(f"-- Generation {current_generation} --")

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # --------------------------------
            # Operators
            # --------------------------------

            DGPLOGGER.debug("    -- Crossing some individuals")
            crossed_individuals = apply_crossover(
                offspring, cx_prob, toolbox.crossover
            )
            DGPLOGGER.info(
                f"    -- Crossed {crossed_individuals} individual pairs."
            )
            DGPLOGGER.debug("    -- Mutating some individuals.")
            mutated_individuals = apply_mutation(
                offspring,
                mut_bias_prob,
                mut_weights_prob,
                toolbox.mutate_bias,
                toolbox.mutate_weights,
            )
            DGPLOGGER.info(
                f"    -- Mutated {mutated_individuals} individuals."
            )

            # --------------------------------
            # Evaluation
            # --------------------------------

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            DGPLOGGER.debug(
                f"    -- Evaluating {len(invalid_ind)} individuals."
            )
            evaluate_population(invalid_ind, toolbox.evaluate)
            DGPLOGGER.info(f"    -- Evaluated {len(invalid_ind)} individuals.")

            population[:] = offspring
            fits = [ind.fitness.values[0] for ind in population]
            finished_generation_summary(current_generation, population)
    except KeyboardInterrupt:
        DGPLOGGER.info("Stopping the algorithm...")

    elapsed_time = time.perf_counter() - time_start
    DGPLOGGER.debug(f"-- End of evolution in {elapsed_time} seconds.")
    population.sort(key=lambda ind: ind.fitness.values, reverse=True)
    finished_algorithm_summary(initial_population, population, tools.selBest)
    test_individual(population[0], model, dataset, fit_train)

    return initial_population[0], population[0]
