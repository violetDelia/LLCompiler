from __future__ import annotations
from glob import glob
from setuptools import setup
from setuptools import Extension
import os
import setuptools
from pybind11.setup_helpers import (
    Pybind11Extension,
    ParallelCompile,
    naive_recompile,
    build_ext,
)
import shutil


import contextlib
import datetime
import glob
import logging
import multiprocessing
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from typing import ClassVar

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop

################################################################################
# Global variables
################################################################################

VERSION = "1.0"
TOP_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, ".setuptools-cmake-build")
WINDOWS = os.name == "nt"
CMAKE = shutil.which("cmake3") or shutil.which("cmake")
BUILDER = shutil.which("ninja")
BUILD_SHARED_LIBS = True
BUILD_TYPE = "Release"
PYBIND_DIR = "/home/lfr/LLCompiler/src/Pybind"

BUILD_LLCOMPILER_TEST = True
STATIC_WINDOWS_RUNTIME = True
BUILD_LLCOMPILER_DOCS = True
BUILD_LLCOMPILER_LOG = True
CMAKE_EXPORT_COMPILE_COMMANDS = True
LLCOMPILER_BUILD_WARNINGS = True


################################################################################
# Pre Check
################################################################################

assert CMAKE, "Could not find cmake in PATH"
assert BUILDER, "Could not find builder in PATH"
################################################################################
# Utilities
################################################################################


@contextlib.contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError(f"Can only cd to absolute path, got: {path}")
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def get_ext_suffix():
    return sysconfig.get_config_var("EXT_SUFFIX")


################################################################################
# CmakeBuild
################################################################################
class CmakeBuild(setuptools.Command):
    """Compiles everything when `python setup.py build` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.

    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """

    user_options: ClassVar[list] = [
        ("jobs=", "j", "Specifies the number of jobs to use with make")
    ]

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        self.set_undefined_options("build", ("parallel", "jobs"))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)

        with cd(CMAKE_BUILD_DIR):
            # configure
            cmake_args = [CMAKE, "-G Ninja"]
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}")
            if BUILD_SHARED_LIBS:
                cmake_args.append("-DBUILD_SHARED_LIBS=ON")
            if BUILD_LLCOMPILER_TEST:
                cmake_args.append("-DBUILD_LLCOMPILER_TEST=ON")
            if STATIC_WINDOWS_RUNTIME:
                cmake_args.append("-DSTATIC_WINDOWS_RUNTIME=ON")
            if BUILD_LLCOMPILER_DOCS:
                cmake_args.append("-DBUILD_LLCOMPILER_DOCS=ON")
            if BUILD_LLCOMPILER_LOG:
                cmake_args.append("-DBUILD_LLCOMPILER_LOG=ON")
            if CMAKE_EXPORT_COMPILE_COMMANDS:
                cmake_args.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
            if LLCOMPILER_BUILD_WARNINGS:
                cmake_args.append("-DLLCOMPILER_BUILD_WARNINGS=ON")
            if "CMAKE_ARGS" in os.environ:
                extra_cmake_args = shlex.split(os.environ["CMAKE_ARGS"])
                # prevent crossfire with downstream scripts
                del os.environ["CMAKE_ARGS"]
                logging.info("Extra cmake args: %s", extra_cmake_args)
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

            build_args = [BUILDER, "install"]
            subprocess.check_call(build_args)


################################################################################
# BuildPy
################################################################################
class BuildPy(setuptools.command.build_py.build_py):
    def run(self):
        return super().run()


################################################################################
# Develop
################################################################################
# class Develop(setuptools.command.develop.develop):
#     def run(self):
#         return super().run()


################################################################################
# BuildExt
################################################################################


ext_modules = []
# could only be relative paths, otherwise the `build` command would fail if you use a MANIFEST.in to distribute your package
# only source files (.cpp, .c, .cc) are needed
source_files = glob.glob("{}/*.cpp".format(PYBIND_DIR), recursive=True)

# If any libraries are used, e.g. libabc.so
include_dirs = [
    "/home/lfr/LLCompiler/include",
    "/home/lfr/LLCompiler/.setuptools-cmake-build/include",
]
library_dirs = ["/home/lfr/LLCompiler/.setuptools-cmake-build/lib"]
# # (optional) if the library is not in the dir like `/usr/lib/`
# # either to add its dir to `runtime_library_dirs` or to the env variable "LD_LIBRARY_PATH"
# # MUST be absolute path
runtime_library_dirs = [
    "/home/lfr/LLCompiler/.setuptools-cmake-build/lib",
]
libraries = ["LLCompiler"]

ext_modules = [
    Pybind11Extension(
        "llcompiler",  # depends on the structure of your package
        source_files,
        # Example: passing in the version to the compiled code
        # include_dirs=include_dirs,
        # library_dirs=library_dirs,
        # runtime_library_dirs=runtime_library_dirs,
        # libraries=libraries,
        language="C++",
        define_macros=[("VERSION_INFO", VERSION)],
    ),
]


################################################################################
# Final
################################################################################

setuptools.setup(
    name="llcompiler",
    version=VERSION,
    cmdclass={
        "cmake_build": CmakeBuild,
        "build_py": BuildPy,
        "build_ext": build_ext,
        # "develop": Develop,
    },
    ext_modules=ext_modules,
)
