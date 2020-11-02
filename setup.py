import sys
import setuptools

LONG_DESC = open("README.md").read()
DOWNLOAD = "https://github.com/lulivi/deep-g-prop/archive/v1.0.0.zip"
REQUIREMENTS = open("requirements/prod.txt").read().splitlines()

setuptools.setup(
    name="DeepGProp",
    version="1.0.0",
    author="Luis Liñán",
    author_email="luislivilla@gmail.com",
    description="Train Multilayer Perceptrons with Genetic Algorithms.",
    long_description_content_type="text/markdown",
    long_description=LONG_DESC,
    license="GPLv3",
    url="https://github.com/lulivi/deep-g-prop",
    download_url=DOWNLOAD,
    classifiers=[
        "Environment :: X11 Applications",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["src"],
    entry_points={"console_scripts": ["dgp=src.deep_g_prop:cli"]},
    python_requires=">=3.6",
    install_requires=REQUIREMENTS,
    test_suite="tests",
)
