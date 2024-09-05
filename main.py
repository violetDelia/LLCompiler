import LLcompiler.Compiler as compiler

# import mlir.ir as ir
import sys
import torchvision


if __name__ == "__main__":
    model = torchvision.models.resnet18()
    compiler = compiler.LLCompiler()
    compiler.importer(model)
