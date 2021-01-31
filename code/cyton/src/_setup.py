"""
Last edit: 16-October-2020
"""

import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="model",
        sources=["model.pyx"],
        # language='c++',
        extra_compile_args=['-O3'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        name="rmodel",
        sources=["rmodel.pyx"],
        # language='c++',
        extra_compile_args=['-O3'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions)
)
