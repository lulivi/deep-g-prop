import sys
import tempfile

from invoke import task, run
from pathlib import Path
from sultan.api import Sultan

ROOT = Path(__file__).parents[1].resolve()
try:
    ASPELL_DIR_PATH = ROOT.joinpath("docs", "aspell").resolve(strict=True)
    REPORT_DIR_PATH = ROOT.joinpath("docs", "report").resolve(strict=True)
    FILTER_DIR_PATH = ROOT.joinpath("docs", "filters").resolve(strict=True)
except FileNotFoundError as e:
    sys.exit("{}: {}".format(e.strerror, e.filename))
REPORT_NAME = "proyecto"


@task
def clean(c):
    """Clean doc construction."""
    print("Cleaning latex and pweave files...")
    patterns = (
        "*.pyc",
        "docs/report/_*",
        "docs/report/figures_pweave",
        "docs/report/*.pyc",
        "docs/report/*.bbl",
        "docs/report/*.blg",
        "docs/report/secciones/*.tex",
        "docs/report/{}.aux".format(REPORT_NAME),
        "docs/report/{}.log".format(REPORT_NAME),
        "docs/report/{}.out".format(REPORT_NAME),
        "docs/report/{}.lof".format(REPORT_NAME),
        "docs/report/{}.lot".format(REPORT_NAME),
        "docs/report/{}.toc".format(REPORT_NAME),
        "docs/report/{}.fls".format(REPORT_NAME),
        "docs/report/{}.fdb_latexmk".format(REPORT_NAME),
        "docs/report/{}.pdf".format(REPORT_NAME),
        "docs/report/{}/".format(REPORT_NAME),
    )

    with Sultan.load(cwd=str(ROOT)) as s:
        for pattern in patterns:
            s.rm("-rf", pattern).run(quiet=True)


@task(clean)
def latex(c):
    """Create tex file."""
    print("Building latex files and figures through pweave ...")
    fig_dir = str(REPORT_DIR_PATH.joinpath("figures_pweave"))
    with Sultan.load(cwd=ROOT) as s:
        for pweave_file in REPORT_DIR_PATH.glob("**/*.texw"):
            tex_file = str(pweave_file.with_suffix(".tex"))
            s.pweave(
                "--format=texminted",
                "--documentation-mode",
                "--figure-directory={}".format(fig_dir),
                "--output={}".format(tex_file),
                "{}".format(str(pweave_file)),
            ).run(quiet=True)


@task(latex)
def plain(c):
    """Create plain documentation from latex."""
    print("Creating plain documentation from latex ...")
    plain_report_path = Path(tempfile.gettempdir()).joinpath(
        "plain_report.txt"
    )
    filters_path = FILTER_DIR_PATH.joinpath("filter.py")
    pandoc_args = [
        str(REPORT_DIR_PATH.joinpath("{}.tex".format(REPORT_NAME))),
        "--output={}".format(plain_report_path),
        "--from=latex",
        "--to=plain",
    ]

    if filters_path.resolve().exists():
        pandoc_args.append("--filter={}".format(str(filters_path)))

    with Sultan.load(cwd=str(REPORT_DIR_PATH)) as s:
        s.pandoc(*pandoc_args).run(quiet=True)


@task(latex)
def pdf(c, bibtex=True):
    """Create pdf file."""
    print("Building pdf through pdflatex ...")
    pdflatex_arg = "-shell-escape {}.tex".format(REPORT_NAME)
    with Sultan.load(cwd=str(REPORT_DIR_PATH), logging=False) as s:
        s.pdflatex(pdflatex_arg).run(quiet=True)
        # Ignore error if there are no citations
        s.bibtex("{}.aux".format(REPORT_NAME)).run(
            quiet=True, halt_on_nonzero=False
        )
        s.pdflatex(pdflatex_arg).run(quiet=True)
        s.pdflatex(pdflatex_arg).run(quiet=True)

