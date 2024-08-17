#llc-opt
C:\coding\LLCompiler\build\tools\llc-opt\llc-opt.exe C:\coding\LLCompiler\test\model_ir\mnist-12.mlir --common-pipeline --dump-pass-pipeline -o=C:\coding\LLCompiler\out.mlir
# onnx-to-mlir
c:\coding\LLCompiler\build\tools\onnx-to-mlir\onnx-to-mlir.exe --log-lever=debug --import-type=onnx_file --input-file=c:\coding\LLCompiler\test\model\mnist-12.onnx --output-file=c:\coding\LLCompiler\test\model_ir\mnist-12.mlir