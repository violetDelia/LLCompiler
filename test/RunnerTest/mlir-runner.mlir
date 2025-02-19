
// RUN: mlir-opt %s -pass-pipeline="builtin.module(convert-linalg-to-loops,convert-scf-to-cf,func.func(convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" | mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module{
    func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
    func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }
    func.func private @printNewline() -> ()
    func.func @main() -> () attributes {entrance, symbol_int_arg_nums = 1 : i64} {
        %c0 = arith.constant {symbol = @c0} 0 : index
        %c1 = arith.constant {symbol = @c1} 1 : index
        %c2 = arith.constant {symbol = @c2} 2 : index
        %1 = arith.index_cast %c2 : index to i64
        %2 = arith.muli %1, %1 : i64
        %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%2 : i64) outs(%alloc : memref<1xf32>) attrs =  {__inplace_operands_attr__ = ["none", "false"]} {
        ^bb0(%in: i64, %out: f32):
        %3 = arith.sitofp %in : i64 to f32
        linalg.yield %3 : f32
        }
        %U = memref.cast %alloc : memref<1xf32> to memref<*xf32>
        // CHECK: Unranked Memref
        // CHECK-SAME: rank = 1 offset = 0 sizes = [1] strides = [1] data =
        // CHECK-NEXT: [4]
        call @printMemrefF32(%U) : (memref<*xf32>) -> ()
        call @printNewline() : () -> ()
    return 
  }
}