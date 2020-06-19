"""Provide types used in varios modules."""

from typing import NamedTuple

import numpy as np


class Proben1Split(NamedTuple):
    """Represent a proben1 split of a partition (train/validation/test).

    :ivar X: data of the split.
    :ivar y: labels of the split.
    :ivar y: labels of the split categorized.

    """

    X: np.ndarray
    y: np.ndarray
    y_cat: np.ndarray


class Proben1Partition(NamedTuple):
    """Represent a proben1 partition.

    :ivar name: name of the partition dataset.
    :ivar nin: number of features.
    :ivar nout: number of classes.
    :ivar trn: train partition.
    :ivar val: validation partition.
    :ivar tst: test partition.

    """

    name: str
    nin: int
    nout: int
    trn: Proben1Split
    val: Proben1Split
    tst: Proben1Split
