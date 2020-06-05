#!/usr/bin/env python3.7

from . import common


def run_search() -> None:
    """Perform the search with a randomized search algorithm."""
    cv = common.RandomizedSearchCV(
        estimator=common.model,
        param_distributions=common.paramgrid,
        scoring="accuracy",
        cv=common.cross_validation,
        verbose=1,
        n_jobs=4,
    )

    result = common.cross_validate("Random Search", cv)
    common.save_result(result)


if __name__ == "__main__":
    run_search()
