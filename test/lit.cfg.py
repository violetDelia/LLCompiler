import os
import sys

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
import lit.util

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'LLC_MLIR'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir',".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(('%PATH%', config.environment['PATH']))
#config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])
llvm_config.use_default_substitutions()
# Tweak the PATH to include the tools dir.
#llvm_config.with_environment("PATH", config.mlir_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.mlir_binary_dir,
    config.mlir_llcompiler_tools_dir,
]
tools = [
    'llc-opt',
    'llc-translate',
    ToolSubst("%PYTHON", config.python_executable, unresolved="ignore"),
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
