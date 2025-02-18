# Submanifold Time Surface Convolutions for Event Streams - keras implementation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

```bash
# install tensorflow (CPU only required - used for data pipelining)
# see https://www.tensorflow.org/install

# install tensorflow-datasets
pip install tensorflow-datasets

# install jax (GPU highly recommended)
# see https://jax.readthedocs.io/en/latest/installation.html

# install keras-nightly (or keras-3.0 if it's been released)
pip install keras-nightly

# c++/cuda ops plus jax wrappers
git clone https://github.com/jackd/jax-stsc-ops.git
pip install -e jax-stsc-ops

# custom datasets
git clone https://github.com/jackd/events-tfds.git
pip install -e events-tfds

# various utility functions
git clone https://github.com/jackd/jk-utils.git
pip install -e jk-utils

# this repo
git clone https://github.com/jackd/stsc.git
pip install -e stsc
```

## Example Usage

```bash
cd stsc
KERAS_BACKEND=jax python scripts/train.py --flagfile=config/ncars.flags --seed=0 --interactive=True
```

Note: some datasets (notably `asl-dvs`) are stored with very large number of files. You may need to increase the open file limit.

```bash
ulimit -n 2048
```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
