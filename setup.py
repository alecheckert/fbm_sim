#!/usr/bin/env python
"""
setup.py

"""
import setuptools

setuptools.setup(
    name="fbm_sim",
    version="1.0",
    packages=setuptools.find_packages(),
    author="Alec Heckert",
    author_email="aheckert@berkeley.edu",
    description="Simulate short fractional Brownian motion trajectories",
)
