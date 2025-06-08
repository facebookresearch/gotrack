# Copyright (c) Meta Platforms, Inc. and affiliates.
cd external/bop_toolkit
pip install -e .
cd ../dinov2
python setup.py install
cd ../..