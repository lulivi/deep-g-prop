#!/usr/bin/env python
"""Module with utility functions."""
import logging

from datetime import datetime
from os import environ
from pathlib import Path
from typing import Dict, List, Set, Union

import pandas as pd  # type: ignore

SRC_DIR = Path(__file__).parent.resolve()


class DatasetNotFoundError(FileNotFoundError):
    """Dataset path does not exist."""


def read_proben1_partition(
    datasets_dir_path: Path, dataset_name: str,
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """Read proben1 dataset partition used in G-Prop paper.

    :param datasets_dir_path: :class:`Path` to dataset directory.
    :param dataset_name: Dataset partition name.
    :raises DatasetNotFoundError: When one of the sets does not exist.
    :returns: Dict of training, validation and test :class:`pd.DataFrame`s with
        their respective partition name.

    """
    try:
        trn_file = (
            datasets_dir_path.joinpath(dataset_name)
            .with_suffix(".trn")
            .resolve(strict=True)
        )
        val_file = (
            datasets_dir_path.joinpath(dataset_name)
            .with_suffix(".val")
            .resolve(strict=True)
        )
        tst_file = (
            datasets_dir_path.joinpath(dataset_name)
            .with_suffix(".tst")
            .resolve(strict=True)
        )
    except FileNotFoundError as error:
        raise DatasetNotFoundError(f"Dataset not found: {error.filename}.")

    df_trn = pd.read_csv(trn_file)
    x_trn = df_trn.drop(columns="class")
    y_trn = df_trn["class"]
    df_val = pd.read_csv(val_file)
    x_val = df_val.drop(columns="class")
    y_val = df_val["class"]
    df_tst = pd.read_csv(tst_file)
    x_tst = df_tst.drop(columns="class")
    y_tst = df_tst["class"]

    return {
        "name": dataset_name,
        "trn": {"X": x_trn, "y": y_trn},
        "val": {"X": x_val, "y": y_val},
        "tst": {"X": x_tst, "y": y_tst},
    }


class PartitionsNotFoundError(FileNotFoundError):
    """No partitions found with the providen name."""


def read_all_proben1_partitions(datasets_dir_path: Path, dataset_name: str):
    """Find all partitions given a dataset name.

    This function will skip files that does not contain numbers at the end of
    the file name, because that means it is the full dataset.

    :param datasets_dir_path: :class:`Path` to the dataset folder.
    :param dataset_name: Dataset name.
    :returns: List with all partitions dictionaries (checkout
        :func:`read_proben1_partition` function).

    """
    partitions_set: Set[str] = set()

    # Obtain all partitions filenames
    for partition in datasets_dir_path.glob(f"{dataset_name}*"):
        if partition.stem[-1].isnumeric():
            partitions_set.add(partition.stem)

    if not partitions_set:
        raise PartitionsNotFoundError(
            f"No partitions found with the name {dataset_name} "
            f"in {str(datasets_dir_path)}"
        )

    df_list: List[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]] = []

    # Obtain all partitions as a list of dictionaries
    for partition_name in sorted(partitions_set):
        df_list.append(
            read_proben1_partition(datasets_dir_path, partition_name)
        )

    return df_list


def validate_level(level: str, default=logging.INFO) -> int:
    """Validate a string logging level.

    :param level: the level string.
    :param default: the default value returned if the ``level`` is not know.
    :returns: the logging level as integer.

    """
    validated_level = logging.getLevelName(level)

    if isinstance(validated_level, int):
        resulting_level = validated_level
    else:
        resulting_level = default

    return resulting_level


def configure_logger(
    name: str = "dgp-logger",
    log_dir: Path = SRC_DIR,
    stream_level: str = "INFO",
) -> logging.Logger:
    """Configure a logger.

    :param name: logger name.
    :param log_dir: path to the directory in which the :class:`FileHandler`
        will output the log.
    :parma stream_level: logging level of the :class:`StreamHandler`.
    :returns: the configured logger.

    """
    datetime_format = "%y%m%d_%H%M%S"
    current_date = datetime.now().strftime(datetime_format)

    logging.root.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_level_var = environ.get("DGP_LOGGER_LEVEL", "") or stream_level
    stream_handler.setLevel(validate_level(stream_level_var, logging.DEBUG))
    stream_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s|%(message)s", datefmt=datetime_format,
        )
    )
    file_handler = logging.FileHandler(log_dir / f"{current_date}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s%(levelname)s|%(message)s",
            datefmt=datetime_format,
        )
    )
    configured_logger = logging.getLogger(name)
    configured_logger.addHandler(stream_handler)
    configured_logger.addHandler(file_handler)

    return configured_logger
