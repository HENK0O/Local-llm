import numpy
from setuptools import Extension, setup


compile_args = ["-O3", "-std=c++17"]

setup(
    ext_modules=[
        Extension(
            "local_llm._native",
            ["local_llm/native/q8.cpp"],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=compile_args,
            language="c++",
            optional=True,
        )
    ]
)
