"""Hyper-parameter optimization with Genetic Algorithms."""
from src.hp_optimization import common


def run_search() -> None:
    """Perform the search with an evolutionary search algorithm."""
    optimizer = common.EvolutionaryAlgorithmSearchCV(
        estimator=common.model,
        params=common.paramgrid,
        scoring="accuracy",
        cv=common.cross_validation,
        verbose=1,
        population_size=10,
        gene_mutation_prob=0.10,
        gene_crossover_prob=0.5,
        tournament_size=3,
        generations_number=5,
        n_jobs=4,
    )

    result = common.cross_validate("Genetic Search", optimizer)
    common.save_result(result)


def main():
    """Call the search run."""
    run_search()


if __name__ == "__main__":
    main()
