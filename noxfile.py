import contextlib
import sys
import tempfile

from functools import partial
from pathlib import Path
from shutil import rmtree
from sys import exit as sysexit
from typing import Callable, Iterator, List, Union

import nox

from nox.command import CommandFailed
from nox.sessions import Session

try:
    from settings import FILTER_DIR_PATH, REPORT_DIR_PATH, REPORT_NAME, ROOT
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from settings import FILTER_DIR_PATH, REPORT_DIR_PATH, REPORT_NAME, ROOT


format_requirements = ["black", "mypy"]
lint_requirements = [
    "pylint",
    "flake8",
    "pycodestyle",
    "isort",
    *format_requirements,
]
test_requirements = [
    "-r",
    "requirements/requirements.txt",
    "-r",
    "requirements/test_requirements.txt",
]
style_target_files = ["noxfile.py", "src", "tests"]
nox.options.sessions = ["tests", "lint"]
nox.options.reuse_existing_virtualenvs = True
latex_already_built = False


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
def get_logger(session: Session):
    """Return the session logger function to call.

    :param session: session from where to obtain the arguments.
    :returns: the session logger, or an empty function.

    """
    return session.log if "verbose" in session.posargs else lambda *args: None


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
        get_logger(session),
    )


@nox.session(name="doc-clean")
def docs_clean(session: Session) -> None:
    """Clean doc construction.
    
    If ``verbose`` is provided in :attr:`Session.posargs`, each file removal
    will be logged. Nothing will print otherwise.

    """
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
        get_logger(session),
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
    global latex_already_built
    if latex_already_built:
        session.log("Skipping latex build as it was already ran.")
        return

    session.log("Building latex files and figures through pweave ...")
    session.install("Pweave==0.30.3")
    fig_dir = str(REPORT_DIR_PATH.joinpath("figures_pweave"))

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
    latex_already_built = True


@nox.session(name="build-plain")
def build_plain(session: Session) -> None:
    """Create plain documentation from latex."""
    build_latex(session)
    session.log("Creating plain documentation from latex ...")
    session.install("panflute==1.12.5")
    plain_report_path = REPORT_DIR_PATH.joinpath(f"{REPORT_NAME}.txt")
    filters_path = FILTER_DIR_PATH.joinpath("filters.py").resolve()
    pandoc_args = [
        str(REPORT_DIR_PATH.joinpath(f"{REPORT_NAME}.tex")),
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
@nox.session(python=["3.6", "3.7", "3.8"])
def tests(session: Session) -> None:
    """Run tests."""
    if session.posargs and session.posargs[0] == "skip-latex":
        session.log("Skipping LaTeX and PDF building for this session.")
    else:
        build_plain(session)
        build_pdf(session)

    session.log("Running tests!")
    session.install(*test_requirements)
    session.install(
        "-r",
        str(ROOT.joinpath("requirements", "requirements.txt")),
        env={"TMPDIR": "/var/tmp/"},
    )
    session.run("pytest", "tests/", "-vvv", silent=False)


# -----------------------------------------------------------------------------
# Format
# -----------------------------------------------------------------------------
@nox.session(name="format")
def apply_format(session: Session) -> None:
    """Apply formating rules to the selected files."""
    session.install("black==19.10b0")
    session.install("isort==4.3.21")
    formating_files = ["noxfile.py", "src", "tests", "docs/filters/"]
    session.run("black", "-l", "79", *formating_files, silent=False)
    session.run("isort", "-rc", *formating_files, silent=False)


# -----------------------------------------------------------------------------
# Lint
# -----------------------------------------------------------------------------
@nox.session()
def lint(session: Session) -> None:
    """Lint the selected files."""
    session.install(*lint_requirements)
    session.run("pylint", *style_target_files)
    session.run("mypy", *style_target_files)
    session.run("flake8", *style_target_files)
    session.run("pycodestyle", *style_target_files)
    session.run("black", "-l", "79", "--check", "--diff", *style_target_files)
    session.run("isort", "-rc", "--check-only", "--diff", *style_target_files)
