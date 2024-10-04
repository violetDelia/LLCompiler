builtin.module attributes  {"builtin.gloabal_layout" = "NCHW"} {
  func.func @main(%0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>  attributes {"entrance"}{
    %1 = "llh.torch_symbolic_int"() {"sym_name" = "s0"} : () -> i64
    %2 = "llh.torch_symbolic_int"() {"sym_name" = "s1"} : () -> i64
    %3 = "llh.torch_symbolic_int"() {"sym_name" = "s2"} : () -> i64
    "llh.symbolic_bind"(%0, %1, %2, %3) {"expressions" = affine_map<()[s0, s1, s2] -> (s0, s1, s2, s2)>} : (tensor<?x?x?x?xf32>, i64, i64, i64) -> ()
    %4 = "llh.add"(%0, %0) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    "llh.symbolic_bind"(%4, %1, %2, %3) {"expressions" = affine_map<()[s0, s1, s2] -> (s0, s1, s2, s2)>} : (tensor<?x?x?x?xf32>, i64, i64, i64) -> ()
    func.return %4 : tensor<?x?x?x?xf32>
  }
}