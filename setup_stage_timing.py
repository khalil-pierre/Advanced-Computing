# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
	Extension(
		"stage_timing",
		["stage_timing.pyx"],
		extra_compile_args=['-fopenmp'],
		extra_link_args=['-fopenmp'],
	)
]

setup(name="stage_timing",
	ext_modules=cythonize(ext_modules))