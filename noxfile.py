"""Automation module to run from documentation builds to tests."""
import contextlib
import sys

from functools import partial
from pathlib import Path
from shutil import rmtree
from typing import Callable, Dict, Iterator, List

import nox  # type: ignore

from nox.command import CommandFailed  # type: ignore
from nox.sessions import Session  # type: ignore

try:
    from settings import (
        FILTER_DIR_PATH,
        REPORT_DIR_PATH,
        REPORT_NAME,
        ROOT,
        REQUIR_DIR_PATH,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from settings import (
        FILTER_DIR_PATH,
        REPORT_DIR_PATH,
        REPORT_NAME,
        REQUIR_DIR_PATH,
        ROOT,
    )

# Configure nox
nox.options.sessions = ["test-docs", "test-sources", "lint"]
nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "venv"

# Globals
form_requirements = ["-r", str(REQUIR_DIR_PATH / "format.txt")]
test_requirements = ["-r", str(REQUIR_DIR_PATH / "tests.txt")]
docs_requirements = ["-r", str(REQUIR_DIR_PATH / "docs.txt")]
prod_requirements = ["-r", str(REQUIR_DIR_PATH / "prod.txt")]
mlp_frameworks_requirements = [
    "-r",
    str(REQUIR_DIR_PATH / "mlp_frameworks.txt"),
]
hp_optimization_requirements = [
    "-r",
    str(REQUIR_DIR_PATH / "hp_optimization.txt"),
]
python_files = [
    "noxfile.py",
    str(FILTER_DIR_PATH / "filters.py"),
    "src",
    "tests",
]


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
def show_help(session: Session, help_dict: Dict[str, str]) -> None:
    """Process the extra arguments for a session.

    :param session: current session.
    :param help: arguments help for the curent session.

    """
    if "help" in session.posargs:
        session.log("=" * 40)
        session.log("Function posargs:")

        for argument, description in help_dict.items():
            session.log(f"\t- {argument}: {description}")

        session.log("=" * 40)
        session.skip()


def remove_files(
    path_list: List[Path], func_logger: Callable = print,
) -> None:
    """Remove the list of paths provided.

    :param path_list: :class:`Path` list to remove.
    :param func_logger: print function to use.

    """

    def remove_file(path: Path, func_logger: Callable = print):
        """Remove provided path.

        :param path: :class:`Path` instance to remove.
        :param func_logger: print function to use.

        """
        try:
            func_logger(f"Trying to remove '{path}'...")
            if path.is_file():
                path.unlink()
            else:
                rmtree(path)
        except FileNotFoundError as error:
            func_logger(
                f"Couldn't remove '{error.filename}'. File/directory not found"
            )
        else:
            func_logger("File/directory removed succesfully")

    list(map(partial(remove_file, func_logger=func_logger), path_list,))


@nox.session(name="py-clean")
def py_clean(session: Session) -> None:
    """Celean python cache files.

    If ``verbose`` is provided in :attr:`Session.posargs`, each file removal
    will be logged. Nothing will print otherwise.

    """
    show_help(session, {"verbose": "Show each deleted file."})
    lggr = session.log if "verbose" in session.posargs else lambda *args: None
    session.log("Cleaning global unwanted files...")

    remove_files(
        [
            *list(ROOT.glob("**/*.py[cod]")),
            *list(ROOT.glob("**/__pycache__/")),
            *list(ROOT.glob("**/.pytest_cache/")),
            *list(ROOT.glob("**/.mypy_cache/")),
            *list(ROOT.glob("**/_minted-*/")),
            *list(ROOT.glob("**/cache/")),
        ],
        lggr,
    )


@nox.session(name="doc-clean")
def docs_clean(session: Session) -> None:
    """Clean doc construction.

    If ``verbose`` is provided in :attr:`Session.posargs`, each file removal
    will be logged. Nothing will print otherwise.

    """
    show_help(session, {"verbose": "Show each deleted file."})
    lggr = session.log if "verbose" in session.posargs else lambda *args: None
    session.log("Cleaning latex and pweave files...")

    remove_files(
        [
            *list(REPORT_DIR_PATH.glob("_*")),
            REPORT_DIR_PATH / "figures_pweave",
            *list(REPORT_DIR_PATH.glob("*.bbl")),
            *list(REPORT_DIR_PATH.glob("*.blg")),
            *list(REPORT_DIR_PATH.glob("secciones/*.tex")),
            REPORT_DIR_PATH / f"{REPORT_NAME}.txt",
            REPORT_DIR_PATH / f"{REPORT_NAME}.aux",
            REPORT_DIR_PATH / f"{REPORT_NAME}.log",
            REPORT_DIR_PATH / f"{REPORT_NAME}.out",
            REPORT_DIR_PATH / f"{REPORT_NAME}.lof",
            REPORT_DIR_PATH / f"{REPORT_NAME}.lot",
            REPORT_DIR_PATH / f"{REPORT_NAME}.toc",
            REPORT_DIR_PATH / f"{REPORT_NAME}.fls",
            REPORT_DIR_PATH / f"{REPORT_NAME}.fdb_latexmk",
            REPORT_DIR_PATH / f"{REPORT_NAME}.pdf",
            REPORT_DIR_PATH / f"{REPORT_NAME}/",
        ],
        lggr,
    )


# -----------------------------------------------------------------------------
# Docs
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def chdir(session: Session, dir_path: Path) -> Iterator[Path]:
    """Temporarily chdir when entering CM and chdir back on exit."""
    orig_dir = Path.cwd()

    session.chdir(str(dir_path))
    try:
        yield dir_path
    finally:
        session.chdir(str(orig_dir))


@nox.session(name="build-latex")
def build_latex(session: Session) -> None:
    """Create tex files from Pweave sources."""
    session.log("Building latex files and figures through pweave ...")
    session.install(*docs_requirements)
    fig_dir = str(REPORT_DIR_PATH / "figures_pweave")

    with chdir(session, ROOT):
        for pweave_file in REPORT_DIR_PATH.glob("**/*.texw"):
            session.run(
                "pweave",
                "--format=texminted",
                "--documentation-mode",
                f"--figure-directory={fig_dir}",
                f"--output={str(pweave_file.with_suffix('.tex'))}",
                str(pweave_file),
                silent=True,
            )

    session.log("Creating plain documentation from latex ...")
    plain_report_path = str(REPORT_DIR_PATH / f"{REPORT_NAME}.txt")
    filters_path = FILTER_DIR_PATH / "filters.py"
    pandoc_args = [
        str(REPORT_DIR_PATH / f"{REPORT_NAME}.tex"),
        f"--output={plain_report_path}",
        "--from=latex",
        "--to=plain",
    ]

    if filters_path.exists():
        pandoc_args.append(f"--filter={str(filters_path)}")

    with chdir(session, REPORT_DIR_PATH):
        session.run("pandoc", *pandoc_args, silent=True, external=True)


@nox.session(name="build-pdf")
def build_pdf(session: Session) -> None:
    """Create pdf file."""
    build_latex(session)
    session.log("Building pdf through pdflatex ...")
    pdflatex_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-shell-escape",
        f"{REPORT_NAME}.tex",
    ]
    with chdir(session, REPORT_DIR_PATH):
        session.run(*pdflatex_cmd, silent=True, external=True)
        # Ignore error if there are no citations
        try:
            session.run(
                "bibtex", f"{REPORT_NAME}.aux", silent=True, external=True
            )
        except CommandFailed:
            pass
        session.run(*pdflatex_cmd, silent=True, external=True)
        session.run(*pdflatex_cmd, silent=True, external=True)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
@nox.session(name="test-docs")
def test_docs(session: Session) -> None:
    """Run the documentation related tests.

    If ``skip-latex`` is provided in :attr:`Session.posargs`, no Pweave/latex
    building will be done before running the tests.

    """
    show_help(session, {"skip-latex": "Skip the latex step."})
    if "skip-latex" in session.posargs:
        session.log("Skipping LaTeX and PDF building for this session.")
    else:
        build_pdf(session)

    session.install(*test_requirements, *docs_requirements)
    session.run("pytest", "-vvv", "-rs", str(ROOT / "tests" / "docs"))


@nox.session(name="test-src", python=["3.7", "3.8"])
def test_src(session: Session) -> None:
    """Run the source code related tests."""
    test_path = (
        session.posargs[0] if session.posargs else str(ROOT / "tests" / "src")
    )
    session.log("Running tests!")
    session.install(
        *test_requirements,
        *mlp_frameworks_requirements,
        *hp_optimization_requirements,
        env={"TMPDIR": "/var/tmp"},
    )
    session.run(
        "pytest",
        test_path,
        f"--cov={str(ROOT / 'src')}",
        "--cov-report=term-missing",
        "-vvv",
        silent=False,
    )


# -----------------------------------------------------------------------------
# Format
# -----------------------------------------------------------------------------
@nox.session(name="format")
def apply_format(session: Session) -> None:
    """Apply formating rules to the selected files."""
    session.install(*form_requirements)
    session.run("black", "-l", "79", *python_files, silent=False)
    session.run(
        "isort", "-rc", "--check-only", "--diff", *python_files, silent=False,
    )


# -----------------------------------------------------------------------------
# Lint
# -----------------------------------------------------------------------------
@nox.session()
def lint(session: Session) -> None:
    """Lint the selected files."""
    session.install(
        *test_requirements,
        *mlp_frameworks_requirements,
        *hp_optimization_requirements,
        *docs_requirements,
        "nox==2020.5.24",
        env={"TMPDIR": "/var/tmp"},
    )
    with chdir(session, ROOT):
        session.run("mypy", *python_files, silent=False)
        session.run("flake8", *python_files, silent=False)
        session.run("pycodestyle", *python_files, silent=False)
        session.run("pydocstyle", *python_files, silent=False)
        session.run(
            "black",
            "-l",
            "79",
            "--check",
            "--diff",
            *python_files,
            silent=False,
        )
        session.run(
            "isort",
            "-rc",
            "--check-only",
            "--diff",
            *python_files,
            silent=False,
        )
        session.run("pylint", *python_files, silent=False)
