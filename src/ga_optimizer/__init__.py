"""Multilayer perceptron optimization via genetic algorithms."""
import random
import time

from typing import List, Tuple

import numpy as np

from deap import tools

from src.common import SEED
from src.dgp_logger import DGPLOGGER
from src.ga_optimizer.toolbox import configure_toolbox
from src.ga_optimizer.types import HiddenLayerInfo, MLPIndividual
from src.ga_optimizer.utils import (
    apply_crossover,
    apply_mutation,
    evaluate_population,
    finished_algorithm_summary,
    finished_generation_summary,
    test_individual,
)
from src.utils import Proben1Partition

__all__ = ["genetic_algorithm"]


# pylint: disable=no-member,too-many-arguments,too-many-locals
def genetic_algorithm(
    hidden_layers_info: List[HiddenLayerInfo],
    dataset: Proben1Partition,
    init_population_size: int,
    max_generations: int,
    cx_prob: float,
    mut_bias_prob: float,
    mut_weights_prob: float,
    mut_neuron_prob: float,
    mut_layer_prob: float,
    fit_train_prob: float,
) -> Tuple[MLPIndividual, MLPIndividual]:
    """Perform optimization with a genetic algorithm.

    :param hidden_layers_info: list of hidden layers basic configuration.
    :param dataset: data to work with.
    :param init_population_size: initial population size.
    :param max_generations: maximun number of generations to run.
    :param cx_prob: probability to mate two individuals.
    :param mut_bias_prob: probability to mutate the individual bias genes.
    :param mut_weights_prob: probability to mutate the individual weights
        genes.
    :param mut_neuron_prob: probability to add/remove a neuron from the model.
    :param mut_layer_prob: probability to add/remove a layer from the model.
    :param fit_train: whether to fit the training data in each evaluation.
    :returns: the first best individual and the final best.

    """
    DGPLOGGER.title(msg="Entering GA")
    np.set_printoptions(precision=5, floatmode="fixed")
    np.random.seed(SEED)
    random.seed(SEED)

    # Configure dataset related variables
    toolbox = configure_toolbox(
        hidden_layers_info,
        dataset,
        {
            "bias": mut_bias_prob,
            "weights": mut_weights_prob,
            "fit": fit_train_prob,
        },
    )

    # --------------------------------
    # Algorithm start
    # --------------------------------
    DGPLOGGER.title(msg="Start the algorithm")
    time_start = time.perf_counter()
    population = toolbox.population(n=init_population_size)

    DGPLOGGER.debug("Evaluate the initial population")
    evaluate_population(population, toolbox.evaluate)

    best_initial_individual = tools.selBest(population, 1)[0]

    initial_population = population[:]
    DGPLOGGER.debug(f"    -- Evaluated {len(population)} individuals")

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in population]
    current_gen_best_fit = max(fits)

    # Variable keeping track of the number of generations
    current_generation = 0

    previous_best_fit = 0
    try:
        # Begin the evolution
        while (
            current_gen_best_fit < 1 and current_generation < max_generations
        ):

            # Check if no score improvement has been made
            if current_generation % 5 == 0:
                if current_gen_best_fit <= previous_best_fit:
                    break

                previous_best_fit = current_gen_best_fit

            # A new generation
            current_generation = current_generation + 1
            DGPLOGGER.title(msg=f"-- Generation {current_generation} --")

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
                offspring, toolbox, mut_neuron_prob, mut_layer_prob
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
            current_gen_best_fit = max(fits)
            finished_generation_summary(current_generation, population)
    except KeyboardInterrupt:
        DGPLOGGER.info("Stopping the algorithm...")

    elapsed_time = time.perf_counter() - time_start
    DGPLOGGER.debug(
        f"-- Finished evolution with the generation {current_generation} in "
        f"{elapsed_time:.2f} seconds."
    )
    finished_algorithm_summary(initial_population, population, tools.selBest)
    test_individual(
        best_initial_individual, dataset, "Best initial individual"
    )
    best_final_individual = tools.selBest(population, 1)[0]
    test_individual(best_final_individual, dataset, "Best final individual")

    return best_initial_individual, best_final_individual
