from setuptools import setup, find_namespace_packages

setup(
    name="2-Object_Detection",
    version="0.1.0",
    description="A set of deep learning object detection models for BDD100k.",
    license="BSD3",
    author="Tiago Almeida",
    author_email="tmr.almeida96@gmail.com",
    python_requires=">=3.6.0",
    url="https://github.com/tmralmeida/bag-of-models/tree/master/CNNs/2-Object_Detection",
    packages=find_namespace_packages(
        exclude=["tests", ".tests", "tests_*", "scripts"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)