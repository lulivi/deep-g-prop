"""Multilayer perceptron testing with Keras."""
from typing import Tuple

import click

from src.common import SEED
from src.dgp_logger import DGPLOGGER
from src.ga_optimizer import genetic_algorithm
from src.proben import Proben1Partition
from src.utils import (
    DatasetNotFoundError,
    print_data_summary,
    read_proben1_partition,
)

DEF_DATASET_NAME = "cancer1"
DEF_INIT_POPULATION = 20
DEF_MAX_GENERATIONS = 10
DEF_NEURONS_RANGE = (2, 20)
DEF_LAYERS_RANGE = (1, 3)
DEF_MUT_CX = 0.5
DEF_MUT_BIAS = 0.2
DEF_MUT_WEIGHTS = 0.75
DEF_MUT_NEURON = 0.3
DEF_MUT_LAYER = 0.3
DEF_CONST_HIDDEN = False
DEF_VERBOSITY = "info"


@click.command()
@click.option(
    "-d",
    "--dataset-name",
    type=click.STRING,
    default=DEF_DATASET_NAME,
    help=(
        "name of the proben1 partition located in src/datasets/. Default: "
        f"'{DEF_DATASET_NAME}'"
    ),
)
@click.option(
    "-ip",
    "--init-pop",
    type=click.INT,
    default=DEF_INIT_POPULATION,
    help=(
        "number of individuals for the first population. Default: "
        f"'{DEF_INIT_POPULATION}'."
    ),
)
@click.option(
    "-mg",
    "--max-gen",
    type=click.INT,
    default=DEF_MAX_GENERATIONS,
    help=f"maximun number of generations. Default: '{DEF_MAX_GENERATIONS}'.",
)
@click.option(
    "-nr",
    "--neurons-range",
    type=(int, int),
    default=DEF_NEURONS_RANGE,
    help=(
        "neurons number range for each hidden layer. Default: "
        f"'{DEF_NEURONS_RANGE}'."
    ),
)
@click.option(
    "-lr",
    "--layers-range",
    type=(int, int),
    default=DEF_LAYERS_RANGE,
    help=f"hidden layers number range. Default: '{DEF_LAYERS_RANGE}'.",
)
@click.option(
    "-cx",
    "--cx-prob",
    type=click.FLOAT,
    default=DEF_MUT_CX,
    help=(
        f"probability for two individuals to mate. Default: '{DEF_MUT_CX}'."
    ),
)
@click.option(
    "-b",
    "--mut-bias",
    type=click.FLOAT,
    default=DEF_MUT_BIAS,
    help=(
        "probability to mutate each individual bias gene. Default: "
        f"'{DEF_MUT_BIAS}'."
    ),
)
@click.option(
    "-w",
    "--mut-weights",
    type=click.FLOAT,
    default=DEF_MUT_WEIGHTS,
    help=(
        "probability to mutate each individual weight gene. Default: "
        f"'{DEF_MUT_WEIGHTS}'."
    ),
)
@click.option(
    "-n",
    "--mut-neurons",
    type=click.FLOAT,
    default=DEF_MUT_NEURON,
    help=(
        "probability to add/remove the last neuron of a random layer for an "
        f"individual. Default: '{DEF_MUT_NEURON}'."
    ),
)
@click.option(
    "-l",
    "--mut-layers",
    type=click.FLOAT,
    default=DEF_MUT_LAYER,
    help=(
        "probability to add/remove the last layer from an individual. "
        f"Default: '{DEF_MUT_LAYER}'."
    ),
)
@click.option(
    "-c",
    "--const-hidden",
    is_flag=True,
    default=DEF_CONST_HIDDEN,
    help=(
        "whether to apply crossover and mutation operators to the hidden "
        f"layers. Default: '{DEF_CONST_HIDDEN}'."
    ),
)
@click.option(
    "-v",
    "--verbosity",
    type=click.Choice(("critical", "info", "debug")),
    default="info",
    help="stream handler verbosity level.",
)
@click.option(
    "-s",
    "--seed",
    type=click.INT,
    default=SEED,
    help="stream handler verbosity level.",
)
# pylint: disable=too-many-arguments,too-many-locals
def cli(
    dataset_name: str,
    init_pop: int,
    max_gen: int,
    neurons_range: Tuple[int, int],
    layers_range: Tuple[int, int],
    cx_prob: float,
    mut_bias: float,
    mut_weights: float,
    mut_neurons: float,
    mut_layers: float,
    const_hidden: bool,
    verbosity: str,
    seed: int,
) -> None:  # noqa: D301
    """Run a genetic algorithm with the chosen settings.

    \f:param dataset_name: data to work with.
    :param init_pop: initial population size.
    :param max_gen: maximun number of generations to run.
    :param neurons_range: max and min values given for the neurons random "
        generator for each layer.
    :param layers_range: max and min values given for the layers random "
        generator.
    :param cx_prob: probability to mate two individuals.
    :param mut_bias: probability to mutate the individual bias genes.
    :param mut_weights: probability to mutate the individual weights genes.
    :param mut_neurons: probability to add/remove a neuron from the model.
    :param mut_layers: probability to add/remove a layer from the model.
    :param const_hidden: ``True`` if the no crossover or mutation can be
        applied to the hidden layers.
    :param verbosity: terminal log verbosity.
    :param seed: random generators seed.

    """
    if neurons_range[0] < 2 or neurons_range[0] > neurons_range[1]:
        print("antonioooo")
        raise click.BadParameter(
            "Wrong neurons range given. It must be inside the range [2, inf). "
            f"Given: '{neurons_range}'.",
            param_hint="--neurons-range",
        )

    if layers_range[0] < 1 or layers_range[0] > layers_range[1]:
        raise click.BadParameter(
            "Wrong hidden layers range given. It must be inside the range [1, "
            f"inf). Given: '{layers_range}'.",
            param_hint="--neurons-range",
        )

    try:
        dataset: Proben1Partition = read_proben1_partition(dataset_name)
    except DatasetNotFoundError as error:
        raise click.BadParameter(
            "Could not find some or any of the partition provided by "
            f"'{dataset_name}'.",
            param_hint="--dataset-name",
        ) from error

    neurons_range_str = ",".join(str(x) for x in neurons_range)
    layers_range_str = ",".join(str(x) for x in layers_range)

    file_name = (
        f"{dataset_name}d_{neurons_range_str}nr_{layers_range_str}lr_"
        f"{init_pop}ip_{max_gen}mg_{cx_prob}cp_{mut_bias}mbp_{mut_weights}mwp_"
        f"{mut_neurons}mnp_{mut_layers}mlp_{const_hidden}c_{seed}s"
    )

    DGPLOGGER.configure_dgp_logger(
        log_stream_level=verbosity, log_file_stem_sufix=file_name
    )

    DGPLOGGER.title(level="debug", msg="Printing dataset sumary:")
    print_data_summary(
        dataset.trn.X, dataset.trn.y, "Train", print_fn=DGPLOGGER.info
    )
    print_data_summary(
        dataset.val.X, dataset.val.y, "Validation", print_fn=DGPLOGGER.info
    )
    print_data_summary(
        dataset.tst.X, dataset.tst.y, "Test", print_fn=DGPLOGGER.info
    )

    DGPLOGGER.title(msg="Selected configuration values")

    DGPLOGGER.info(f"-- Dataset name: {dataset.name}")
    DGPLOGGER.info(f"-- Initial population size: {init_pop}")
    DGPLOGGER.info(f"-- Maximun number of generations: {max_gen}")
    DGPLOGGER.info(f"-- Neurons per hidden layer range: {neurons_range}")
    DGPLOGGER.info(f"-- Hidden layers number range: {layers_range}")
    DGPLOGGER.info(f"-- Cossover probability: {cx_prob}")
    DGPLOGGER.info(f"-- Bias gene mutation probability: {mut_bias}")
    DGPLOGGER.info(f"-- Weights gene mutation probability: {mut_weights}")
    DGPLOGGER.info(f"-- Neuron mutation probability: {mut_neurons}")
    DGPLOGGER.info(f"-- Layer mutation probability: {mut_layers}")
    DGPLOGGER.info(f"-- Constant hidden layers: {const_hidden}")
    DGPLOGGER.info(f"-- Seed: {seed}")

    genetic_algorithm(
        dataset=dataset,
        init_population_size=init_pop,
        max_generations=max_gen,
        neurons_range=neurons_range,
        layers_range=layers_range,
        cx_prob=cx_prob,
        mut_bias_prob=mut_bias,
        mut_weights_prob=mut_weights,
        mut_neuron_prob=mut_neurons,
        mut_layer_prob=mut_layers,
        const_hidden_layers=const_hidden,
        seed=seed,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
