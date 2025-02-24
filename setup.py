from __future__ import annotations
from glob import glob
from setuptools import setup
from setuptools import Extension
import os
import setuptools
from pybind11.setup_helpers import (
    Pybind11Extension,
    build_ext,
)
import shutil


import contextlib
import glob
import logging
import multiprocessing
import os
import shlex
import shutil
import subprocess
import sysconfig
from typing import ClassVar

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop


################################################################################
# Build options
################################################################################
BUILD_LLCOMPILER_TEST = False
STATIC_WINDOWS_RUNTIME = True
BUILD_LLCOMPILER_DOCS = True
BUILD_LLCOMPILER_LOG = True
CMAKE_EXPORT_COMPILE_COMMANDS = True
LLCOMPILER_BUILD_WARNINGS = True
DEVELOPER_MODE = False
BUILD_SHARED_LIBS = True
BUILD_TYPE = "Release"
################################################################################
# Global variables
################################################################################
VERSION = "0.0.1"
TOP_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, "build")
WINDOWS = os.name == "nt"
CMAKE = shutil.which("cmake3") or shutil.which("cmake")
BUILDER = shutil.which("ninja")
PYBIND_DIR = os.path.join(TOP_DIR, "src", "Pybind")
INSTALL_DIR = os.path.join(TOP_DIR, "install")
INCLUDE_DIRS = [os.path.join(INSTALL_DIR, "include")]
LIBRARY_DIRS = [os.path.join(INSTALL_DIR, "lib")]
RUNTIME_LIBRARY_DIRS = [os.path.join(INSTALL_DIR, "lib")]
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

    def bool_arg(seld, option: str):
        option_value = globals()[option]
        arg = f"-D{option}=" + ("ON" if option_value else "OFF")
        return arg

    def run(self):
        os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)

        with cd(CMAKE_BUILD_DIR):
            # configure
            cmake_args = [CMAKE, "-G Ninja"]
            cmake_args.append(f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}")
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}")
            cmake_args.append(self.bool_arg("BUILD_SHARED_LIBS"))
            cmake_args.append(self.bool_arg("BUILD_LLCOMPILER_TEST"))
            cmake_args.append(self.bool_arg("STATIC_WINDOWS_RUNTIME"))
            cmake_args.append(self.bool_arg("BUILD_LLCOMPILER_DOCS"))
            cmake_args.append(self.bool_arg("BUILD_LLCOMPILER_LOG"))
            cmake_args.append(self.bool_arg("CMAKE_EXPORT_COMPILE_COMMANDS"))
            cmake_args.append(self.bool_arg("LLCOMPILER_BUILD_WARNINGS"))
            cmake_args.append(self.bool_arg("DEVELOPER_MODE"))
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
# BuildExt
################################################################################


class BuildExt(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command("cmake_build")
        return super().run()


################################################################################
# ext_modules
################################################################################
ext_modules = []
source_files = glob.glob("{}/*.cpp".format(PYBIND_DIR), recursive=True)
libraries = ["LLCompiler"]
ext_modules = [
    Pybind11Extension(
        "llcompiler_",
        source_files,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=RUNTIME_LIBRARY_DIRS,
        libraries=libraries,
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
        "build_ext": BuildExt,
    },
    ext_modules=ext_modules,
)
