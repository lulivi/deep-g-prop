from invoke import task, run
from . import docs


@task(pre=[docs.plain, docs.pdf])
def go(c):
    """Run tests."""
    print("Running unittests...")
    run("pytest tests/ -v", pty=True)
