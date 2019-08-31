import panflute as pf


def remove_environments(elem, doc):
    """Remove code and math blocks."""
    if (
        isinstance(elem, pf.CodeBlock)
        or isinstance(elem, pf.Math)
        or (isinstance(elem, pf.Str) and (elem.text == "image"))
    ):
        return []


def main(doc=None):
    return pf.run_filter(remove_environments, doc=doc)


if __name__ == "__main__":
    main()
