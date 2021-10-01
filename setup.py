# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
import setuptools

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

dev_requires = ["tox", "black", "flake8", "flake8-black", "numpy", "torch_testing"]

docs_requires = [
    "sphinx==3.3.1",
    "recommonmark",
    "sphinx_rtd_theme",
    "sphinx_markdown_builder",
    "ipython",  # required for Pygments
    "nbsphinx",
    "sphinxcontrib_katex",
    "pytorch_sphinx_theme @ git+https://github.com/pytorch/pytorch_sphinx_theme.git",
]

setuptools.setup(
    name="torchlip",
    version="0.0.1",
    author=", ".join(
        [
            "Mathieu SERRURIER",
            "Franck MAMALET",
            "Thibaut BOISSIN",
            "Mikaël CAPELLE",
            "Justin PLAKOO",
        ]
    ),
    author_email=", ".join(
        [
            "mathieu.serrurier@irt-saintexupery.com",
            "franck.mamalet@irt-saintexupery.com",
            "thibaut.boissin@irt-saintexupery.com",
            "mikael.capelle@irt-saintexupery.com",
            "justin.plakoo@irt-saintexupery.com",
        ]
    ),
    description="PyTorch implementation for k-Lipschitz layers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deel-ai/torchlip",
    packages=setuptools.find_namespace_packages(include=["deel.*"]),
    install_requires=["numpy", "inflection"],
    license="MIT",
    extras_require={"dev": dev_requires, "doc": docs_requires},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    # Requirements:
    install_requires=[
        "torch_testing",
        "inflection",
        "numpy",
        "torch",
    ],
    python_requires=">=3.6",
)
