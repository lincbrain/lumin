from setuptools import setup, find_packages

package = "lsm"
version = "0.1"

setup(
    name=package,
    version=version,
    description="light-sheet microscopy segmentation evaluation framework",
    packages=find_packages(),
)
