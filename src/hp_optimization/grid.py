#!/usr/bin/env python3.7

from . import common


def run_search() -> None:
    """Perform the search with grid search algorithm."""
    cv = common.GridSearchCV(
        estimator=common.model,
        param_grid=common.paramgrid,
        scoring="accuracy",
        cv=common.cross_validation,
        verbose=1,
        n_jobs=4,
    )

    result = common.cross_validate("Grid Search", cv)
    common.save_result(result)


if __name__ == "__main__":
    run_search()
