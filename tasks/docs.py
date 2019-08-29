import sys

from invoke import task
from pathlib import Path
from sultan.api import Sultan

REPO_ROOT = Path(__file__).parents[1]
default_report_name = "proyecto"


@task
def update_word_list(c, extra_word_list_file="extra_words", ext="txt"):
    """Update aspell extra dictionary from word list."""
    print("Updating aspell extra words list...")
    try:
        aspell_dicts_path = REPO_ROOT.joinpath("docs", "aspell").resolve()
        extra_word_list_path = aspell_dicts_path.joinpath(
            "{}.{}".format(extra_word_list_file, ext)
        ).resolve()
    except FileNotFoundError as e:
        sys.exit("Error: {}".format(e))

    aspell_extra_dict_path = aspell_dicts_path.joinpath(
        "{}.rws".format(extra_word_list_file)
    )

    with Sultan.load(cwd=str(aspell_dicts_path)) as s:
        s.cat(str(extra_word_list_path)).pipe().aspell(
            "--lang=es", "create", "master", str(aspell_extra_dict_path)
        ).run(quiet=True)


@task
def clean(c, report_name=default_report_name):
    """Clean doc construction."""
    print("Cleaning ...")
    patterns = (
        "*.pyc",
        "docs/report/_*",
        "docs/report/figures",
        "docs/report/*.pyc",
        "docs/report/*.bbl",
        "docs/report/*.blg",
        "docs/report/{}.aux".format(report_name),
        "docs/report/{}.log".format(report_name),
        "docs/report/{}.out".format(report_name),
        "docs/report/{}.lof".format(report_name),
        "docs/report/{}.lot".format(report_name),
        "docs/report/{}.toc".format(report_name),
        "docs/report/{}.fls".format(report_name),
        "docs/report/{}.fdb_latexmk".format(report_name),
        "docs/report/{}.tex".format(report_name),
        "docs/report/{}.pdf".format(report_name),
        "docs/report/{}/".format(report_name),
    )

    with Sultan.load() as s:
        for pattern in patterns:
            s.rm("-rf {}".format(pattern)).run(quiet=True)


@task(clean)
def latex(c, report_name=default_report_name):
    """Create tex file."""
    print("Building latex file and figures through pweave ...")
    try:
        report_name_path = Path("docs", "report", report_name)
    except FileNotFoundError:
        sys.exit("{} not found.".format(report_name_path))

    with Sultan.load(cwd="docs/report") as s:
        s.pweave("-f", "texminted {}.texw".format(report_name)).run(quiet=True)


@task(latex)
def pdf(c, report_name=default_report_name, bibtex=True):
    """Create pdf file."""
    print("Building pdf through pdflatex ...")
    pdflatex_arg = "-shell-escape {}.tex".format(report_name)
    with Sultan.load(cwd="docs/report") as s:
        s.pdflatex(pdflatex_arg).run(quiet=True)
        if bibtex:
            s.bibtex("{}.aux".format(report_name)).run(quiet=True)
        s.pdflatex(pdflatex_arg).run(quiet=True)
        s.pdflatex(pdflatex_arg).run(quiet=True)

