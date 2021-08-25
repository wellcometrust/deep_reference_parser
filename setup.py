import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# Load data from the__versions__.py module. Change version, etc in
# that module, and it will be automatically populated here.

about = {}  # type: dict
version_path = os.path.join(here, "deep_reference_parser", "__version__.py")
with open(version_path, "r") as f:
    exec(f.read(), about)

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name=about["__name__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    license=["__license__"],
    packages=[
        "deep_reference_parser",
        "deep_reference_parser/prodigy",
        "deep_reference_parser/io",
    ],
    package_dir={"deep_reference_parser": "deep_reference_parser"},
    package_data={
        "deep_reference_parser": [
            f"configs/{about['__splitter_model_version__']}.ini",
            f"configs/{about['__parser_model_version__']}.ini",
            f"configs/{about['__splitparser_model_version__']}.ini",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "spacy<2.2.0",
        "pandas",
        "tensorflow==2.5.1",
        "keras==2.2.5",
        "keras-contrib @ https://github.com/keras-team/keras-contrib/tarball/5ffab172661411218e517a50170bb97760ea567b",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz#egg=en_core_web_sm==2.1.0",
        "sklearn",
        "sklearn_crfsuite",
        "matplotlib",
    ],
    tests_require=["pytest", "pytest-cov"],
)
