# make onnx
C:\LLCompiler\build\tools\onnx-to-mlir\onnx-to-mlir.exe --import-type=onnx_file --input-file=C:\LLCompiler\test\models\mnist-12.onnx --output-file=C:\LLCompiler\test\model_ir\mnist-12.mlir
#llh to tosa
c:\LLCompiler\build\tools\llc-opt\llc-opt.exe c:\LLCompiler\test\model_ir\mnist-12.mlir --inline --convert-llh-to-tosa --dump-pass-pipeline --debug-only=dialect-conversion