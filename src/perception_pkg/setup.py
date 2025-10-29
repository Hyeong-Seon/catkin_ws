from setuptools import find_packages, setup

setup(
    name="perception_pkg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
    ],
)
