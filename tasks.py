from invoke import run
from invoke import task


@task
def tests(c):
    """Run tests."""
    print("Running tests...")
    run("pytests tests/")


@task
def clean_docs(c):
    """Clean doc construction."""
    print("Cleaning ...")
    patterns = [
        "*.log",
        "docs/report/_*",
        "docs/report/figures",
        "docs/report/*.aux",
        "docs/report/*.log",
        "docs/report/*.out",
        "docs/report/pweave_example.tex",
        "docs/report/*.pdf",
        "*.pyc",
        "docs/report/*.pyc",
        "docs/report/*.bbl",
        "docs/report/*.blg",
    ]
    for pattern in patterns:
        c.run("rm -rf {}".format(pattern))


@task
def build_latex_report(c):
    """Create tex file."""
    print("Building latex file and figures through pweave ...")
    command = "cd docs/report && pweave -f texminted report.texw"
    run(command, hide=False, warn=True)


@task
def build_pdf_report(c):
    """Create pdf file."""
    print("Building pdf through pdflatex ...")
    command = (
        "cd docs/report "
        "&& pdflatex -shell-escape report.tex "
        "&& bibtex pweave_example.aux "
        "&& pdflatex -shell-escape pweave_example.tex "
        "&& pdflatex -shell-escape pweave_example.tex"
    )
    run(command, hide=False, warn=True)


@task
def build(c):
    build_latex_report(c)
    build_pdf_report(c)
