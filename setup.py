import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="multilayer-perceptron",
    version="1.0.0",
    url="https://github.com/Mertaami/multilayer-perceptron",
    author="Kagamino",
    author_email="martin@lehoux.net",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
    ],
    packages=["MultiLayerPerceptron"],
    install_requires=["numpy"],
    include_package_data=True
)