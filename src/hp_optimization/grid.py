"""Hyper-parameter optimization with Grid search."""
try:
    from src.hp_optimization import common
except ImportError:  # pragma: no cover
    from sys import path as syspath
    from pathlib import Path

    syspath.append(str(Path(__file__).resolve().parents[2]))

    from src.hp_optimization import common


def run_search() -> None:
    """Perform the search with grid search algorithm."""
    optimizer = common.GridSearchCV(
        estimator=common.model,
        param_grid=common.paramgrid,
        scoring="accuracy",
        cv=common.cross_validation,
        verbose=1,
        n_jobs=4,
    )

    result = common.cross_validate("Grid Search", optimizer)
    common.save_result(result)


def main():
    """Call the search run."""
    run_search()


if __name__ == "__main__":  # pragma: no cover
    main()
