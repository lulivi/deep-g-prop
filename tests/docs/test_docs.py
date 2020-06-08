"""Tests related to the documentation."""
import subprocess
import sys
import unittest

from settings import ASPELL_DIR_PATH, REPORT_DIR_PATH, REPORT_NAME, ROOT


class ReportTests(unittest.TestCase):
    """Run some report tests."""

    def test_report_spelling(self):
        """Run aspell and check if there is any misspelled word."""
        process0 = subprocess.Popen(
            ["cat", str(self.plain_report_path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
        )
        process1 = subprocess.Popen(
            ["aspell", "--lang=es_ES", "list"],
            cwd=ROOT,
            stdin=process0.stdout,
            stdout=subprocess.PIPE,
        )
        process2 = subprocess.Popen(
            ["aspell", "--lang=en_US", "list", *self.aspell_extra_args],
            cwd=ROOT,
            stdin=process1.stdout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        output = process2.communicate()[0].split()
        self.assertListEqual(output, [], "List of wrong words is not empty.")

    @unittest.skip("Not finished documentation")
    def test_pdf_pages_number(self):
        """Check if report has the minimun number of pages."""
        number_of_pages = int(
            subprocess.run(
                [
                    "qpdf",
                    "--show-npages",
                    str(REPORT_DIR_PATH / "proyecto.pdf"),
                ],
                universal_newlines=True,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
        )

        self.assertGreaterEqual(
            number_of_pages, 50, "Minimun report pages not reached."
        )

    @classmethod
    def setUpClass(cls):
        """Create proper environment to check report spelling."""
        cls.plain_report_path = REPORT_DIR_PATH / f"{REPORT_NAME}.txt"

        if not cls.plain_report_path.exists():
            sys.exit(
                "Latex plain report not found. Ensure to run 'nox -e "
                "build-plain' before running the tests."
            )

        cls.pdf_report_path = REPORT_DIR_PATH / "proyecto.pdf"

        if not cls.pdf_report_path.exists():
            sys.exit(
                "Latex PDF report not found. Ensure to run 'nox -e "
                "build-docs' before running the tests."
            )

        aspell_extra_dict = ASPELL_DIR_PATH / "personal.aspell.en.pws"
        cls.aspell_extra_args = []

        if aspell_extra_dict.exists():
            cls.aspell_extra_args.append(
                "--add-extra-dicts={}".format(aspell_extra_dict)
            )


if __name__ == "__main__":
    unittest.main()
