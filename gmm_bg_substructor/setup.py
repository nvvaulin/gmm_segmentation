from distutils.core import setup,Extension
from Cython.Build import cythonize


extensions = [
    Extension('pybgs',
        ['pybgs.pyx'],
        extra_compile_args=["-std=c++14"])
]


setup(
	name = "pybgs",
	include_dirs = ['/usr/include/opencv/', '/usr/include/opencv2/core', '/usr/include/opencv2/highgui'],
	ext_modules = cythonize(extensions)
	
)

