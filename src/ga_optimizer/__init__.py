"""Multilayer perceptron optimization via genetic algorithms."""
import random
import time

from typing import Tuple

import numpy as np

from deap import tools

from src.dgp_logger import DGPLOGGER
from src.ga_optimizer.toolbox import configure_toolbox
from src.ga_optimizer.types import MLPIndividual
from src.ga_optimizer.utils import (
    apply_crossover,
    apply_mutation,
    evaluate_population,
    finished_algorithm_summary,
    finished_generation_summary,
    test_individual,
)
from src.utils import Proben1Partition

__all__ = ["genetic_algorithm", "MLPIndividual", "test_individual"]


# pylint: disable=no-member,too-many-arguments,too-many-locals
def genetic_algorithm(
    dataset: Proben1Partition,
    init_population_size: int,
    max_generations: int,
    neurons_range: Tuple[int, int],
    layers_range: Tuple[int, int],
    cx_prob: float,
    mut_bias_prob: float,
    mut_weights_prob: float,
    mut_neuron_prob: float,
    mut_layer_prob: float,
    const_hidden_layers: bool,
    seed: int,
) -> Tuple[MLPIndividual, MLPIndividual]:
    """Perform optimization with a genetic algorithm.

    :param dataset: data to work with.
    :param init_population_size: initial population size.
    :param max_generations: maximun number of generations to run.
    :param neurons_range: max and min values given for the neurons random "
        generator for each layer.
    :param layers_range: max and min values given for the layers random "
        generator.
    :param cx_prob: probability to mate two individuals.
    :param mut_bias_prob: probability to mutate the individual bias genes.
    :param mut_weights_prob: probability to mutate the individual weights
        genes.
    :param mut_neuron_prob: probability to add/remove a neuron from the model.
    :param mut_layer_prob: probability to add/remove a layer from the model.
    :param const_hidden_layers: ``True`` if the no crossover or mutation can be
        applied to the hidden layers.
    :param seed: seed for random number generators.
    :returns: the first best individual and the final best.

    """
    DGPLOGGER.title(msg="Entering GA")
    np.set_printoptions(precision=5, floatmode="fixed")
    np.random.seed(seed)
    random.seed(seed)

    # Configure dataset related variables
    toolbox = configure_toolbox(
        dataset,
        neurons_range=neurons_range,
        layers_range=layers_range,
        mut_bias_prob=mut_bias_prob,
        mut_weights_prob=mut_weights_prob,
        const_hidden_layers=const_hidden_layers,
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

    current_generation = 0
    current_gen_best_fit = max([ind.fitness for ind in population])
    previous_best_fit = None

    try:
        # Begin the evolution
        while (
            current_gen_best_fit.values != (0.0, 1.0, 1.0)
            and current_generation < max_generations
        ):

            # Check if no score improvement has been made
            if current_generation % 10 == 0:
                if (
                    previous_best_fit
                    and current_gen_best_fit <= previous_best_fit
                ):
                    DGPLOGGER.info(
                        "The fitness has not improved in 10 generations:\n"
                        f"\tPrevious best fit: {previous_best_fit}\n"
                        f"\tCurrent best fit: {current_gen_best_fit}\n"
                        "Exiting..."
                    )
                    break

                previous_best_fit = current_gen_best_fit

            # A new generation
            current_generation = current_generation + 1
            DGPLOGGER.title(msg=f"-- Generation {current_generation} --")

            # Select the best individuals for the offspring
            best_population_individuals = tools.selBest(
                population, int(len(population) / 2)
            )
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, best_population_individuals))

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

            # Replace the worst individuals from the previous population with
            # the mutated ones. In other words, create the new offspring
            # from the previous population best individual plus the mutated
            # ones
            population = best_population_individuals + offspring
            current_gen_best_fit = tools.selBest(population, 1)[0].fitness

            finished_generation_summary(
                current_generation, population, current_gen_best_fit.values
            )
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
