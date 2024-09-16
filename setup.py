from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import os

BASE_DIR = os.path.dirname(__file__)
PYBIND_DIR = os.path.join(BASE_DIR, "LLcompiler", "pybind")
os.chdir(BASE_DIR)
ext_modules = []
try:
    from pybind11.setup_helpers import (
        Pybind11Extension,
        ParallelCompile,
        naive_recompile,
        build_ext,
    )

    ParallelCompile(
        "NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile, default=4
    ).install()

    # could only be relative paths, otherwise the `build` command would fail if you use a MANIFEST.in to distribute your package
    # only source files (.cpp, .c, .cc) are needed
    source_files = glob("{}/*.cpp".format(PYBIND_DIR), recursive=True)

    # If any libraries are used, e.g. libabc.so
    include_dirs = []
    library_dirs = []
    # # (optional) if the library is not in the dir like `/usr/lib/`
    # # either to add its dir to `runtime_library_dirs` or to the env variable "LD_LIBRARY_PATH"
    # # MUST be absolute path
    runtime_library_dirs = []
    libraries = []

    print(source_files)
    
    ext_modules = [
        Pybind11Extension(
            "example",  # depends on the structure of your package
            source_files,
            # Example: passing in the version to the compiled code
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=runtime_library_dirs,
            libraries=libraries,
        ),
    ]
except ImportError:
    pass
print(ext_modules)
setup(
    name="test_aaaaaaaaa",  # used by `pip install`
    version="0.0.1",
    description="xxx",
    setup_requires=["pybind11"],
    install_requires=["pybind11"],
    python_requires='>=3.8',
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
