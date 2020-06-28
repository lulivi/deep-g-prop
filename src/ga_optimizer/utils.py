"""Util functions used in the GA."""
import random

from typing import Callable, Tuple

import numpy as np

from deap import base

from src.dgp_logger import DGPLOGGER
from src.ga_optimizer.toolbox import individual_evaluator
from src.ga_optimizer.types import MLPIndividual
from src.types import Proben1Partition
from src.utils import print_table


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

    for crossover_index, (child1, child2) in enumerate(
        zip(population[::2], population[1::2])
    ):
        if random.random() < cx_prob and child1.can_mate(child2):
            crossed_individuals += 1
            DGPLOGGER.debug(
                f"    Applying crossover for the {crossover_index} couple "
                f"({crossover_index * 2}, {crossover_index * 2+1}))."
            )
            crossover_fn(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    return crossed_individuals


def apply_mutation(
    population: list,
    toolbox: base.Toolbox,
    mut_neuron_prob: float,
    mut_layer_prob: float,
) -> int:
    """Mutate the population with probabilities.

    :param population: list of individuals.
    :param toolbox: object where the mutation operators are defined.
    :param mut_neuron_prob: probability to mute a neuron (append/pop).
    :param mut_layer_prob: probability to mute a layer (append/pop).
    :returns: the number of individuals mutated.

    """
    mutated_individuals = 0

    for index, mutant in enumerate(population):
        mut_bias_genes = mut_weights_genes = neuron_diff = layer_diff = 0

        mut_bias_genes = toolbox.mutate_bias(mutant)
        mut_weights_genes = toolbox.mutate_weights(mutant)
        del mutant.fitness.values

        if random.random() < mut_neuron_prob:
            neuron_diff = toolbox.mutate_neuron(mutant)

        if random.random() < mut_layer_prob:
            layer_diff = toolbox.mutate_layer(mutant)

        if mut_bias_genes or mut_weights_genes:
            mutated_individuals += 1
            DGPLOGGER.debug(
                f"    For individual {index}:\n"
                f"        {mut_bias_genes} mutated bias genes\n"
                f"        {mut_weights_genes} mutated weights genes\n"
                f"        {neuron_diff} neuron changes\n"
                f"        {layer_diff} layer changes\n"
            )

    return mutated_individuals


def finished_generation_summary(current_generation: int, population: list):
    """Print a summary of the current generation.

    :param current_generation: index of the generation.
    :param population: list of individuals of the current generation.

    """
    fits = np.array([ind.fitness.values for ind in population])
    table = [
        ["Statistic", "F2 score", "Accuracy error %"],
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
        "headers": ["Index", "F2 score", "Accuracy error %"],
        "print_fn": DGPLOGGER.debug,
        "floatfmt": ("i", ".5f", ".2f"),
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
    final_pop_neurons_table.append(["Std", *final_pop_layer_list.std(0)])
    DGPLOGGER.debug("Final population layer neurons statistics:")
    print_table(
        final_pop_neurons_table,
        print_fn=DGPLOGGER.debug,
        floatfmt=(None, *[".2f" for idx in range(max_layer_ind)]),
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
    val_scores_no_train = individual_evaluator(
        individual,
        dataset.trn,
        dataset.val,
        multi_class=dataset.nout > 2,
        fit_train_prob=0.0,
    )
    val_scores_train = individual_evaluator(
        individual,
        dataset.trn,
        dataset.val,
        multi_class=dataset.nout > 2,
        fit_train_prob=1.0,
    )
    tst_scores_no_train = individual_evaluator(
        individual,
        dataset.trn,
        dataset.tst,
        multi_class=dataset.nout > 2,
        fit_train_prob=0.0,
    )
    tst_scores_train = individual_evaluator(
        individual,
        dataset.trn,
        dataset.tst,
        multi_class=dataset.nout > 2,
        fit_train_prob=1.0,
    )
    individual_table = [
        ["Validation (no train)", *val_scores_no_train],
        ["Validation (train)", *val_scores_train],
        ["Test (no train)", *tst_scores_no_train],
        ["Test (train)", *tst_scores_train],
    ]
    print_table(
        individual_table,
        print_fn=DGPLOGGER.info,
        floatfmt=(None, ".5f", ".2f"),
        headers=["Partition", "F2 score", "Accuracy error %"],
    )
    return tst_scores_train
