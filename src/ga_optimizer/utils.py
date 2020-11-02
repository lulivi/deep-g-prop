"""Util functions used in the GA."""
import random

from typing import Callable, Tuple

import numpy as np

from deap import base

from src.dgp_logger import DGPLOGGER
from src.ga_optimizer.toolbox import individual_evaluator
from src.ga_optimizer.types import MLPIndividual
from src.proben import Proben1Partition
from src.utils import print_table


def evaluate_population(population: list, evaluate_fn: Callable) -> None:
    """Evaluate the population.

    Apply the evaluation method to every individual of the population.

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

    Apply crossover to ``cx_prob`` percentage of the population in pairs if
    both individuals can mate toguether.

    :param population: list of individuals.
    :param cx_prob: mate probability.
    :param crossover_fn: function to mate two individuals.
    :returns: number of individual pairs mated.

    """
    crossed_individuals = 0

    for crossover_index, (child1, child2) in enumerate(
        zip(population[::2], population[1::2])
    ):
        if random.random() < cx_prob and child1.can_mate(child2):
            crossed_individuals += 1
            cx_pt, layer_index = crossover_fn(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            DGPLOGGER.debug(
                f"    Applying crossover for the {crossover_index} couple "
                f"({crossover_index * 2}, {crossover_index * 2+1})).\n"
                f"        Crossed from neuron {cx_pt[0]} to {cx_pt[1]} in "
                f"layer {layer_index}"
            )

    return crossed_individuals


def apply_mutation(
    population: list,
    toolbox: base.Toolbox,
    mut_neurons_prob: float,
    mut_layers_prob: float,
) -> int:
    """Mutate the population with probabilities.

    Mutate weights and bias elements for every individual. Then randomly mutate
    neuron count for ``mut_neurons_prob`` individuals and layer count for
    ``mut_layers_prob`` individuals.

    :param population: list of individuals.
    :param toolbox: object where the mutation operators are defined.
    :param mut_neurons_prob: probability to mute a neuron (append/pop).
    :param mut_layers_prob: probability to mute a layer (append/pop).
    :returns: the number of individuals mutated.

    """
    mutated_individuals = 0

    for index, mutant in enumerate(population):
        mut_bias_genes = mut_weights_genes = neuron_diff = layer_diff = 0

        mut_bias_genes = toolbox.mutate_bias(mutant)
        mut_weights_genes = toolbox.mutate_weights(mutant)
        mutated_individuals += 1
        del mutant.fitness.values

        # Ensure that we don't modify the hidden layers if they are constant
        if not mutant.constant_hidden_layers:
            if random.random() < mut_neurons_prob:
                neuron_diff = toolbox.mutate_neuron(mutant)

            if random.random() < mut_layers_prob:
                layer_diff = toolbox.mutate_layer(mutant)

        DGPLOGGER.debug(
            f"    For individual {index}:\n"
            f"        {mut_bias_genes} mutated bias genes\n"
            f"        {mut_weights_genes} mutated weights genes\n"
            f"        {neuron_diff} neuron changes\n"
            f"        {layer_diff} layer changes\n"
        )

    return mutated_individuals


def finished_generation_summary(
    current_generation: int, population: list, best_fit: tuple
):
    """Print a summary of the current generation.

    :param current_generation: index of the generation.
    :param population: list of individuals of the current generation.

    """
    fits = np.array([ind.fitness.values for ind in population])
    table = [
        ["Statistic", "Accuracy error %", "Neuron/Layer score", "F2 score"],
        ["Max", *fits.max(0)],
        ["Avg", *fits.mean(0)],
        ["Min", *fits.min(0)],
        ["Std", *fits.std(0)],
        ["Best", *best_fit],
    ]
    DGPLOGGER.info(f"    Summary of generation {current_generation}:")
    print_table(table, DGPLOGGER.info, floatfmt=(None, ".2f", ".2f", ".5f"))


def finished_algorithm_summary(
    initial_population, final_population, best_individual_selector
):
    """Print a summary of the hole process.

    Show last population hidden layer info, first and last population fitness.

    :param initial_population: list of initial individuals.
    :param final_population: list of final individuals.
    :param best_individual_selector: function to select the best individual.

    """
    initial_pop_table = []
    final_pop_table = []
    final_pop_neurons_table = []
    final_pop_ind_layer_list = [
        [layer.config["units"] for layer in ind.layers[:-1]]
        for ind in final_population
    ]
    max_layer_ind = max(map(len, final_pop_ind_layer_list))
    final_pop_layer_list = np.array(
        [
            layer_list + [0] * (max_layer_ind - len(layer_list))
            for layer_list in final_pop_ind_layer_list
        ]
    )
    table_common_attributes = {
        "headers": [
            "Index",
            "Accuracy error %",
            "Neuron/Layer score",
            "F2 score",
        ],
        "print_fn": DGPLOGGER.debug,
        "floatfmt": ("i", ".2f", ".2f", ".5f"),
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

    final_pop_neurons_table.append(
        [
            "Statistic",
            *[f"Hidden layer {idx}" for idx in range(max_layer_ind)],
        ]
    )
    final_pop_neurons_table.append(["Max", *final_pop_layer_list.max(0)])
    final_pop_neurons_table.append(["Mean", *final_pop_layer_list.mean(0)])
    final_pop_neurons_table.append(["Min", *final_pop_layer_list.min(0)])
    final_pop_neurons_table.append(["Std", *final_pop_layer_list.std(0)])
    DGPLOGGER.debug("Final population layer neurons statistics:")
    print_table(
        final_pop_neurons_table,
        print_fn=DGPLOGGER.debug,
        floatfmt=(None, ".2f", ".2f", ".5f"),
    )


def test_individual(
    individual: MLPIndividual,
    dataset: Proben1Partition,
    text: str = "Individual",
) -> Tuple[float, float]:
    """Test an individual with the validation and test data.

    :param individual: current individual to evaluate.
    :param dataset: data to work with.
    :param text: text to print before the table.

    """
    DGPLOGGER.info(f"{text} weights")
    DGPLOGGER.info(str(individual))
    DGPLOGGER.info(
        "Predicting the validation and test data with the best individual."
    )
    val_scores = individual_evaluator(
        individual, dataset.trn, dataset.val, multi_class=dataset.nout > 2,
    )
    tst_scores = individual_evaluator(
        individual, dataset.trn, dataset.tst, multi_class=dataset.nout > 2,
    )
    individual_table = [
        ["Validation", *val_scores],
        ["Test", *tst_scores],
    ]
    print_table(
        individual_table,
        print_fn=DGPLOGGER.info,
        floatfmt=(None, ".2f", ".2f", ".5f"),
        headers=[
            "Partition",
            "Accuracy error %",
            "Neuron/layer score",
            "F2 score",
        ],
    )
    return tst_scores
