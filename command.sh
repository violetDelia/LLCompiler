# onnx-to-mlir
c:\coding\LLCompiler\build\tools\onnx-to-mlir\onnx-to-mlir.exe --log-lever=debug --import-type=onnx_file --input-file=c:\coding\LLCompiler\test\model\mnist-12.onnx --output-file=c:\coding\LLCompiler\test\model_ir\mnist-12.mlir --log-root=C:\coding\LLCompiler\log
#llc-opt
 '/home/lfr/LLCompiler/build/tools/llc-opt/llc-opt' --dump-pass-pipeline -o=C:\coding\LLCompiler\opt_out.mlir --log-lever=debug  --log-root=C:\coding\LLCompiler\log --mlir-print-ir-tree-dir=C:\coding\LLCompiler\mlir-ir-dir --mlir-print-ir-after-all -common-pipeline C:\coding\LLCompiler\test\model_ir\mnist-12.mlir 
#llc-opt only-compiler
C:\coding\LLCompiler\build\tools\llc-opt\llc-opt.exe C:\coding\LLCompiler\test\model_ir\mnist-12.mlir --dump-pass-pipeline -o=C:\coding\LLCompiler\opt_out.mlir --log-lever=debug  --log-root=C:\coding\LLCompiler\log --mlir-print-ir-tree-dir=C:\coding\LLCompiler\mlir-ir-dir-only-compiler --mlir-print-ir-after-all --common-pipeline="only-compiler=true"
# llc-translate
c:\coding\LLCompiler\build\tools\llc-translate\llc-translate.exe c:\coding\LLCompiler\opt_out.mlir --mlir-to-llvmir -o=c:\coding\LLCompiler\llvmir.ll
#
c:\coding\LLCompiler\build\third_party\llvm-project\llvm\bin\llc.exe c:\coding\LLCompiler\llvmir.ll
c:\coding\LLCompiler\build\third_party\llvm-project\llvm\bin\llvm-dis.exe c:\coding\LLCompiler\llvmir.s
