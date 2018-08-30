#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:38:12 2018

@author: Soumya
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("world_view_cython.pyx")
)
