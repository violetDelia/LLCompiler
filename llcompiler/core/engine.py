from llcompiler_.entrance import ExecutionEngine
import torch.fx
from typing import Any, Union, List, Dict


class ExecutionEngine:

    def __init__(self, ExecutionEngine):
        self.engine = ExecutionEngine

    def debug_info(self):
        self.engine.debug_info()

    def run(self, *args) -> Any:
        print("Running")
        return args[0]

    def __call__(self, *args) -> Any:
        return self.run(*args)
