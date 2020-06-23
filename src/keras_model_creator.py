"""Create Keras models to be the base for the GA optimization."""

from pathlib import Path
from pprint import pformat
from typing import List

import click
import keras

from settings import MODELS_DIR_PATH
from src.dgp_logger import DGPLOGGER
from src.types import Proben1Partition
from src.utils import DatasetNotFoundError, read_proben1_partition


class HiddenLayerSequence(click.ParamType):
    """Custom click type to match a hidden layer sequence."""

    name = "hidden layer sequence"

    def convert(self, value, param, ctx):
        """Conver from string to a list of ints."""
        try:
            return list(map(int, value.split()))
        except ValueError:
            self.fail(
                f"{value!r} is not a valid hidden layer sequence", param, ctx
            )


@click.command()
@click.argument("model-name", type=click.STRING, required=True)
@click.argument("dataset-name", type=click.STRING, required=True)
@click.argument(
    "hidden-layers-size",
    type=HiddenLayerSequence(),
    default=[5],
    required=True,
)
@click.option(
    "--verbosity",
    type=click.Choice(("INFO", "DEBUG")),
    default="INFO",
    help="stream handler verbosity level.",
)
def cli(
    model_name: Path,
    dataset_name: str,
    hidden_layers_size: List[int],
    verbosity: str,
) -> None:
    """Create a Keras model and save it compiled.

    :param model_name: name of the model to save.

    :param dataset_name: name of the proben1 partition to use the dimensions
        from.

    :param hidden-layers-size: sequence of ints for the hidden layers.

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

    # Configure log file handler
    DGPLOGGER.configure_dgp_logger(
        log_stream_level=verbosity, log_file_stem_sufix=Path(__file__).stem
    )

    DGPLOGGER.title(msg="Creating Keras model...")
    model = keras.models.Sequential()
    common_attributes = {
        "use_bias": True,
        "activation": "relu",
        "kernel_initializer": keras.initializers.Zeros(),
        "bias_initializer": keras.initializers.Zeros(),
    }
    model.add(
        keras.layers.Dense(
            units=hidden_layers_size[0],
            input_shape=(dataset.nin,),
            name="HiddenLayer0",
            **common_attributes,
        )
    )

    for index, layer_units in enumerate(hidden_layers_size[1:]):
        model.add(
            keras.layers.Dense(
                units=layer_units,
                name=f"HiddenLayer{index+1}",
                **common_attributes,
            )
        )

    common_attributes["activation"] = "softmax"
    model.add(
        keras.layers.Dense(
            units=dataset.nout, name="OutputLayer", **common_attributes,
        )
    )

    weights = model.get_weights()
    DGPLOGGER.debug("Choosen layers:")
    DGPLOGGER.debug(str([(layer.shape, layer.dtype) for layer in weights]))
    model.summary(print_fn=DGPLOGGER.debug)
    DGPLOGGER.debug(pformat(model.get_weights()))
    DGPLOGGER.title(msg="Compiling Keras model...")
    model.compile(
        optimizer="sgd", loss="binary_crossentropy",
    )

    # serialize model to HDF5
    hidden_layers_str = "_".join(map(str, hidden_layers_size))
    model_path = str(
        MODELS_DIR_PATH
        / f"{dataset.name[:-1]}_{hidden_layers_str}_{model_name}_model.h5"
    )
    model.save(model_path)
    DGPLOGGER.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
