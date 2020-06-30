"""Multilayer perceptron testing with Keras."""
from logging import DEBUG
from typing import List

import click

from src.dgp_logger import DGPLOGGER
from src.ga_optimizer import HiddenLayerInfo, genetic_algorithm
from src.types import Proben1Partition
from src.utils import (
    DatasetNotFoundError,
    print_data_summary,
    read_proben1_partition,
)


class HiddenLayerInfoList(click.ParamType):
    """Custom click type to match a hidden layer sequence."""

    name = "hidden layer info sequence"

    @staticmethod
    def _get_layer_info(layer_info_str):
        """Return a hidden layer info object from a string."""
        neurons_str, trainable_str = layer_info_str.split()
        return HiddenLayerInfo(
            int(neurons_str), click.BOOL(trainable_str.strip()),
        )

    def convert(self, value, param, ctx):
        """Conver from string to a :class:`List[HiddenLayerInfo]`."""
        try:
            return list(map(self._get_layer_info, value.split(",")))
        except ValueError:
            self.fail(
                f"{value!r} is not a valid hidden layer sequence. It sould be "
                "'<num_neurons> <traineable>[, <neurons> <traineable> [...]]'"
                ". For example: '4 True, 2 False, 6 True'",
                param,
                ctx,
            )


@click.command()
@click.option(
    "--dataset-name",
    type=click.STRING,
    default="cancer1",
    help="name of the proben1 partition located in src/datasets/",
)
@click.option(
    "--hidden-layers-info",
    type=HiddenLayerInfoList(),
    default="3 True",
    help=(
        'sequence of hidden layer configuration in the form of "4 True, 2'
        ' False" to have two hidden layers: the first one trainable with 4 '
        "neurons and the second one non-trainable with 2."
    ),
)
@click.option(
    "--init-population-size",
    type=click.INT,
    default=50,
    help="number of individuals for the first population.",
)
@click.option(
    "--max-generations",
    type=click.INT,
    default=300,
    help="maximun number of generations.",
)
@click.option(
    "--cx-prob",
    type=click.FLOAT,
    default=0.5,
    help="probability for two individuals to mate.",
)
@click.option(
    "--mut-bias-prob",
    type=click.FLOAT,
    default=0.2,
    help="probability to mutate each individual bias gene.",
)
@click.option(
    "--mut-weights-prob",
    type=click.FLOAT,
    default=0.75,
    help="probability to mutate each individual weight gene.",
)
@click.option(
    "--mut-neuron-prob",
    type=click.FLOAT,
    default=0.3,
    help=(
        "probability to add/remove the last neuron of a random layer for an "
        "individual."
    ),
)
@click.option(
    "--mut-layer-prob",
    type=click.FLOAT,
    default=0.3,
    help="probability to add/remove the last layer from an individual.",
)
@click.option(
    "--fit-train-prob",
    type=click.FLOAT,
    default=0.3,
    help=(
        "probability to fit the training data for each individual in each "
        "evaluation."
    ),
)
@click.option(
    "--verbosity",
    type=click.Choice(("INFO", "DEBUG")),
    default="INFO",
    help="stream handler verbosity level.",
)
# pylint: disable=too-many-arguments
def cli(
    dataset_name: str,
    hidden_layers_info: List[HiddenLayerInfo],
    init_population_size: int,
    max_generations: int,
    cx_prob: float,
    mut_bias_prob: float,
    mut_weights_prob: float,
    mut_neuron_prob: float,
    mut_layer_prob: float,
    fit_train_prob: float,
    verbosity: str,
) -> None:
    """Run a genetic algorithm with the chosen settings.

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

    :param fit_train_prob: probability to fit the training data for each
        individual in each evaluation.

    :param verbosity: terminal log verbosity.

    """
    # Load the dataset
    try:
        dataset: Proben1Partition = read_proben1_partition(dataset_name)
    except DatasetNotFoundError as error:
        DGPLOGGER.critical(
            "There was an error when reading proben1 partition "
            f"'{dataset_name}': {error}"
        )
        raise click.BadParameter(
            "Could not find some or any of the partition provided by "
            f"'{dataset_name}'.",
            param_hint="--dataset-name",
        ) from error

    hidden_layers_str = "_".join(
        [
            f"{hidden.neurons}{'t' if hidden.trainable else 'n'}"
            for hidden in hidden_layers_info
        ]
    )
    file_name = f"{dataset_name}_{hidden_layers_str}"

    if mut_neuron_prob > 0.0:
        file_name += "_neur"

    if mut_layer_prob > 0.0:
        file_name += "_lay"

    if fit_train_prob > 0.0:
        file_name += "_fit"

    # Configure log file handler
    DGPLOGGER.configure_dgp_logger(
        log_stream_level=verbosity, log_file_stem_sufix=file_name
    )

    # Data summary
    DGPLOGGER.title(level=DEBUG, msg="Printing dataset sumary:")
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
    DGPLOGGER.info(
        f"-- Hidden layer sequence configuration: {hidden_layers_info}"
    )
    DGPLOGGER.info(f"-- Dataset name: {dataset.name}")
    DGPLOGGER.info(f"-- Initial population size: {init_population_size}")
    DGPLOGGER.info(f"-- Maximun number of generations: {max_generations}")
    DGPLOGGER.info(f"-- Cossover probability: {cx_prob}")
    DGPLOGGER.info(f"-- Bias gene mutation probability: {mut_bias_prob}")
    DGPLOGGER.info(f"-- Weights gene mutation probability: {mut_weights_prob}")
    DGPLOGGER.info(f"-- Neuron mutation probability: {mut_neuron_prob}")
    DGPLOGGER.info(f"-- Layer mutation probability: {mut_layer_prob}")
    DGPLOGGER.info(
        f"-- Fit train before predicting probability: {fit_train_prob}"
    )

    # Call the genetic algorithm
    genetic_algorithm(
        hidden_layers_info=hidden_layers_info,
        dataset=dataset,
        init_population_size=init_population_size,
        max_generations=max_generations,
        cx_prob=cx_prob,
        mut_bias_prob=mut_bias_prob,
        mut_weights_prob=mut_weights_prob,
        mut_neuron_prob=mut_neuron_prob,
        mut_layer_prob=mut_layer_prob,
        fit_train_prob=fit_train_prob,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
