#llc-opt
C:\coding\LLCompiler\build\tools\llc-opt\llc-opt.exe C:\coding\LLCompiler\test\model_ir\mnist-12.mlir --dump-pass-pipeline -o=C:\coding\LLCompiler\out.mlir --log-lever=debug  --log-root=C:\coding\LLCompiler\log --mlir-print-ir-tree-dir=C:\coding\LLCompiler\mlir-ir-dir --mlir-print-ir-after-all -common-pipeline
# onnx-to-mlir
c:\coding\LLCompiler\build\tools\onnx-to-mlir\onnx-to-mlir.exe --log-lever=debug --import-type=onnx_file --input-file=c:\coding\LLCompiler\test\model\mnist-12.onnx --output-file=c:\coding\LLCompiler\test\model_ir\mnist-12.mlir --log-root=C:\coding\LLCompiler\log

C:\coding\LLCompiler\build\tools\llc-opt\llc-opt.exe c:\coding\LLCompiler\third_party\llvm-project\mlir\test\Dialect\Tensor\bufferize.mlir --one-shot-bufferize="dialect-filter=tensor,bufferization copy-before-write unknown-type-conversion=identity-layout-map"  -split-input-file -o=c:\coding\LLCompiler\test\mlir_temp_file\bufferize.mlir