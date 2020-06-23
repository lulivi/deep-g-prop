"""Multilayer perceptron testing with Keras."""
from logging import DEBUG
from pathlib import Path
from pprint import pformat

import click
import keras

from src.dgp_logger import DGPLOGGER
from src.ga_optimizer import genetic_algorithm
from src.types import Proben1Partition
from src.utils import (
    DatasetNotFoundError,
    print_data_summary,
    read_proben1_partition,
)


@click.command()
@click.argument("model-path", type=Path, required=True)
@click.option(
    "--dataset-name",
    type=click.STRING,
    default="cancer1",
    help="name of the proben1 partition located in src/datasets/",
)
@click.option(
    "--fit-train",
    type=click.BOOL,
    default=False,
    help="whether to train the model when evaluating each individual.",
)
@click.option(
    "--verbosity",
    type=click.Choice(("INFO", "DEBUG")),
    default="INFO",
    help="stream handler verbosity level.",
)
def cli(
    model_path: Path, dataset_name: str, fit_train: bool, verbosity: str
) -> None:
    """Load <model-path> Keras model and optimize it with genetic algorithms.

    :param model_path: path to the ``h5`` model.

    :param dataset_name: name of the proben1 partition.

    :param fit_train: whether to fit the model before predicting labels.

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

    if (
        not model_path.exists()
        or dataset.name[:-1] not in model_path.stem
        or model_path.suffix != ".h5"
    ):
        DGPLOGGER.critical(
            "There was an error when lading the provided model from "
            f"'{str(model_path)}'"
        )
        raise click.BadParameter(
            "Model path does not exist, it is not compatible with the dataset "
            "provided, or it is not a '.h5' file.",
            param_hint="MODEL_PATH",
        )

    # Configure log file handler
    DGPLOGGER.configure_dgp_logger(
        log_stream_level=verbosity, log_file_stem_sufix=Path(__file__).stem
    )

    # Load the keras model
    DGPLOGGER.title(msg=f"Loading Keras model from {str(model_path)}")
    model = keras.models.load_model(str(model_path))
    model.summary(print_fn=DGPLOGGER.debug)
    DGPLOGGER.debug(pformat(model.get_weights()))

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

    # Call the genetic algorithm
    genetic_algorithm(model, dataset, fit_train=fit_train)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
