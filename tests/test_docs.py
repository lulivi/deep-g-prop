import unittest
import tempfile

from PyPDF2 import PdfFileReader
from pathlib import Path
from sultan.api import Sultan

REPO_ROOT = Path(__file__).parents[1]


class ReportTests(unittest.TestCase):
    """Run some report tests."""

    def test_report_spelling(self):
        "Run aspell and check if there is any misspelled word."
        with Sultan.load(cwd=REPO_ROOT, logging=False) as s:
            result = (
                s.cat(self.output_report_path)
                .pipe()
                .aspell("--lang=es_ES", "list", *self.aspell_extra_arg)
                .run(quiet=True, halt_on_nonzero=False)
            )

        self.assertEqual(
            result.stdout, [], "List of wrong words is not empty."
        )

    def test_pdf_pages_number(self):
        """Check if report has the minimun number of pages."""
        with open(REPO_ROOT.joinpath("docs", "report", "proyecto.pdf")) as f:
            pdf = PdfFileReader(f)
            number_of_pages = pdf.getNumPages()

        self.assertGreaterEqual(
            number_of_pages, 50, "Minimun report pages not reached."
        )

    @classmethod
    def setUpClass(cls):
        """Create proper environment to check report spelling.

        Firstly, search for the latex report. If it does not exist not, compile
        it with Pyweave.
        Secondly, check if there are any filters for pandoc conversion.
        Then, find out if an aspell personal dictionary exists.
        Finally, convert the report to plain text, to check the spelling.

        """
        try:
            report_path = REPO_ROOT.joinpath(
                "docs", "report", "proyecto.tex"
            ).resolve()

            if not report_path.exists():
                raise FileNotFoundError
        except FileNotFoundError:
            with Sultan.load(cwd=str(REPO_ROOT)) as s:
                s.pipenv("run", "inv", "docs.latex").run(quiet=True)

        try:
            filter_path = REPO_ROOT.joinpath(
                "docs", "filters", "filters.py"
            ).resolve()

            if not filter_path.exists():
                raise FileNotFoundError
        except FileNotFoundError:
            filter_path = None

        try:
            aspell_extra_dict_path = REPO_ROOT.joinpath(
                "docs", "aspell", "extra_words.rws"
            ).resolve()

            if not aspell_extra_dict_path.exists():
                raise FileNotFoundError
        except FileNotFoundError:
            cls.aspell_extra_arg = []
        else:
            cls.aspell_extra_arg = [
                "--add-extra-dicts={}".format(aspell_extra_dict_path)
            ]

        cls.output_report_path = Path(tempfile.gettempdir()).joinpath(
            "filtered_report.txt"
        )

        pandoc_args = [
            "--output={}".format(str(cls.output_report_path)),
            "--from=latex",
            "--to=plain",
        ]

        if filter_path:
            pandoc_args.append("--filter={}".format(str(filter_path)))

        pandoc_args.append(str(report_path))

        with Sultan.load(cwd=str(REPO_ROOT.joinpath("docs", "report"))) as s:
            s.pandoc(*pandoc_args).run(quiet=True)


if __name__ == "__main__":
    unittest.main()
