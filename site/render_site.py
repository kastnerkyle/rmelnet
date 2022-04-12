#!/usr/bin/env python3
"""Generate forms for human evaluation."""

from jinja2 import FileSystemLoader, Environment
import os
import glob
import numpy as np


def main():
    """Main function."""
    loader = FileSystemLoader(searchpath="./templates")
    env = Environment(loader=loader)
    template = env.get_template("site.html.jinja2")
    html = template.render(
        page_title="Samples site",
    )
    print(html)


if __name__ == "__main__":
    main()
