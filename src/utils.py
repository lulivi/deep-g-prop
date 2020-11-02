#!/usr/bin/env python
"""Module with utility functions."""
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd

from tabulate import tabulate

from settings import PROBEN1_DIR_PATH
from src.proben import Proben1Partition, Proben1Split


class DatasetNotFoundError(FileNotFoundError):
    """Dataset path does not exist."""


def read_proben1_partition(
    dataset_name: str, datasets_dir_path: Path = PROBEN1_DIR_PATH,
) -> Proben1Partition:
    """Read proben1 dataset partition used in G-Prop paper.

    :param dataset_name: Dataset partition name.
    :param datasets_dir_path: :class:`Path` to dataset directory.
    :raises DatasetNotFoundError: When one of the sets does not exist.
    :returns: Dict of training, validation and test :class:`pd.DataFrame`s with
        their respective partition name.

    """
    try:
        trn_file = (
            (datasets_dir_path / dataset_name)
            .with_suffix(".trn")
            .resolve(strict=True)
        )
        val_file = (
            (datasets_dir_path / dataset_name)
            .with_suffix(".val")
            .resolve(strict=True)
        )
        tst_file = (
            (datasets_dir_path / dataset_name)
            .with_suffix(".tst")
            .resolve(strict=True)
        )
    except FileNotFoundError as error:
        raise DatasetNotFoundError(f"Split not found: {error.filename}.")

    trn_data = pd.read_csv(trn_file)
    trn_labels = trn_data.pop("class")
    trn_labels_cat = pd.get_dummies(trn_labels)

    val_data = pd.read_csv(val_file)
    val_labels = val_data.pop("class")
    val_labels_cat = pd.get_dummies(val_labels)

    tst_data = pd.read_csv(tst_file)
    tst_labels = tst_data.pop("class")
    tst_labels_cat = pd.get_dummies(tst_labels)

    return Proben1Partition(
        dataset_name,
        len(trn_data.columns),
        pd.concat((trn_labels, val_labels, tst_labels)).nunique(),
        Proben1Split(
            trn_data.to_numpy(),
            trn_labels.to_numpy(),
            trn_labels_cat.to_numpy(),
        ),
        Proben1Split(
            val_data.to_numpy(),
            val_labels.to_numpy(),
            val_labels_cat.to_numpy(),
        ),
        Proben1Split(
            tst_data.to_numpy(),
            tst_labels.to_numpy(),
            tst_labels_cat.to_numpy(),
        ),
    )


class PartitionsNotFoundError(FileNotFoundError):
    """No partitions found with the providen name."""


def read_all_proben1_partitions(
    dataset_name: str, datasets_dir_path: Path = PROBEN1_DIR_PATH
) -> List[Proben1Partition]:
    """Find all partitions given a dataset name.

    This function will skip files that does not contain numbers at the end of
    the file name, because that means it is the full dataset.

    :param dataset_name: Dataset name.
    :param datasets_dir_path: :class:`Path` to the dataset folder.
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

    df_list: List[Proben1Partition] = []

    # Obtain all partitions as a list of dictionaries
    for partition_name in sorted(partitions_set):
        df_list.append(
            read_proben1_partition(partition_name, datasets_dir_path)
        )

    return df_list


def print_table(table: List[List[str]], print_fn=print, **kwargs) -> None:
    """Print data in a table format.

    :param table: the data to print.
    :param print_fn: print function used to output the table.

    """
    headers = kwargs.pop("headers", "firstrow")
    tablefmt = kwargs.pop("tablefmt", "simple")
    colalign = kwargs.pop(
        "colalign", ["center" for column in range(len(table[0]))]
    )
    print_fn(
        tabulate(
            tabular_data=table,
            headers=headers,
            tablefmt=tablefmt,
            colalign=colalign,
            **kwargs,
        )
    )


def print_data_summary(
    data: np.ndarray, labels: np.ndarray, name: str = "", print_fn=print
):
    """Print a summary of the dataset provided.

    :param data: dataset data.
    :param labels: dataset labels.
    :param name: name to display in the summary.

    """
    print_fn(f"Data summary: {name}")
    print_fn(f"data.shape = {data.shape}")
    print_fn(f"labels.shape = {labels.shape}")
    total_elements = len(labels)
    print_fn("Class distribution:")

    for category, count in np.asarray(np.unique(labels, return_counts=True)).T:
        print_fn(f"\t{category} - {count} ({count / total_elements:.2f})")
