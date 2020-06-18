"""Apply panflute filters to the project documentation."""
from typing import Optional

import panflute as pf


def remove_environments(element, doc) -> Optional[list]:
    """Remove code and math blocks."""
    # pylint: disable=unused-argument
    if isinstance(element, (pf.CodeBlock, pf.Math)) or (
        isinstance(element, pf.Str) and (element.text == "image")
    ):
        return []
    return None


def main():
    """Run the panflute filter."""
    return pf.run_filter(remove_environments, doc=None)


if __name__ == "__main__":
    main()
