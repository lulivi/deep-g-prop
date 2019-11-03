import sys
import unittest
import tempfile

from PyPDF2 import PdfFileReader
from pathlib import Path
from sultan.api import Sultan

ROOT = Path(__file__).parents[1].resolve()
try:
    ASPELL_DIR_PATH = ROOT.joinpath("docs", "aspell").resolve(strict=True)
    REPORT_DIR_PATH = ROOT.joinpath("docs", "report").resolve(strict=True)
except FileNotFoundError as e:
    sys.exit("{}: {}".format(e.strerror, e.filename))


class ReportTests(unittest.TestCase):
    """Run some report tests."""

    def test_report_spelling(self):
        """Run aspell and check if there is any misspelled word."""
        with Sultan.load(cwd=ROOT, logging=False) as s:
            result = (
                s.cat(str(self.plain_report_path))
                .pipe()
                .aspell("--lang=es_ES", "list")
                .pipe()
                .aspell("--lang=en_US", "list", *self.aspell_extra_args)
                .run(quiet=True, halt_on_nonzero=False)
            )

        self.assertEqual(
            result.stdout, [], "List of wrong words is not empty."
        )

    @unittest.skip("Not finished documentation")
    def test_pdf_pages_number(self):
        """Check if report has the minimun number of pages."""
        pdf = PdfFileReader(str(self.pdf_report_path))
        number_of_pages = pdf.getNumPages()

        self.assertGreaterEqual(
            number_of_pages, 50, "Minimun report pages not reached."
        )

    @classmethod
    def setUpClass(cls):
        """Create proper environment to check report spelling."""
        cls.plain_report_path = Path(tempfile.gettempdir()).joinpath(
            "plain_report.txt"
        )

        if not cls.plain_report_path.exists():
            sys.exit("Latex report not found. Can not continue.")

        cls.pdf_report_path = REPORT_DIR_PATH.joinpath("proyecto.pdf")

        if not cls.pdf_report_path.exists():
            sys.exit("PDF report not found. Can not continue.")

        aspell_extra_dict = ASPELL_DIR_PATH.joinpath("personal.aspell.en.pws")
        cls.aspell_extra_args = []

        if aspell_extra_dict.exists():
            cls.aspell_extra_args.append(
                "--add-extra-dicts={}".format(aspell_extra_dict)
            )


if __name__ == "__main__":
    unittest.main()
