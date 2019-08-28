from invoke import task
from pathlib import Path
from sultan.api import Sultan

REPO_ROOT = Path(__file__).parents[1]


@task
def run(c):
    """Run tests."""
    print("Running tests...")
    with Sultan.load(str(REPO_ROOT)) as s:
        result = s.pytest("tests/", "-v").run(
            quiet=True, halt_on_nonzero=False
        )
    for i in result.stdout:
        print(i)
