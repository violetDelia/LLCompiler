"builtin.module"() ({
  "llh.symbolic_int"() <{sym_name = "c88"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c92"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c96"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c100"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c200"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c7"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
  "func.func"() <{arg_attrs = [{func.input_symbol_0 = "c3", func.input_symbol_1 = "c3", func.input_symbol_2 = "c7", func.input_symbol_3 = "c7"}, {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c100", func.input_symbol_3 = "c100"}], function_type = (tensor<3x3x7x7xf32>, tensor<200x3x100x100xf32>) -> (tensor<200x3x88x88xf32>, tensor<3x3x7x7xf32>, tensor<200x3x100x100xf32>, tensor<200x3x96x96xf32>, tensor<200x3x92x92xf32>), sym_name = "main"}> ({
  ^bb0(%arg0: tensor<3x3x7x7xf32>, %arg1: tensor<200x3x100x100xf32>):
    "llh.encoding_bind"(%arg1) <{encoding = #llh.encoding<shapes = @c200, @c3, @c100, @c100>}> : (tensor<200x3x100x100xf32>) -> ()
    "llh.encoding_bind"(%arg0) <{encoding = #llh.encoding<shapes = @c3, @c3, @c7, @c7>}> : (tensor<3x3x7x7xf32>) -> ()
    %0 = "llh.transpose"(%arg1) <{perms = array<i64: 0, 2, 3, 1>}> : (tensor<200x3x100x100xf32>) -> tensor<200x100x100x3xf32>
    %1 = "llh.transpose"(%arg0) <{perms = array<i64: 0, 2, 3, 1>}> : (tensor<3x3x7x7xf32>) -> tensor<3x7x7x3xf32>
    %2 = "llh.conv"(%0, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NHWC>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<200x100x100x3xf32>, tensor<3x7x7x3xf32>) -> tensor<200x96x96x3xf32>
    %3 = "llh.transpose"(%2) <{perms = array<i64: 0, 3, 1, 2>}> : (tensor<200x96x96x3xf32>) -> tensor<200x3x96x96xf32>
    "llh.encoding_bind"(%3) <{encoding = #llh.encoding<shapes = @c200, @c3, @c96, @c96>}> : (tensor<200x3x96x96xf32>) -> ()
    %4 = "llh.transpose"(%3) <{perms = array<i64: 0, 2, 3, 1>}> : (tensor<200x3x96x96xf32>) -> tensor<200x96x96x3xf32>
    %5 = "llh.transpose"(%arg0) <{perms = array<i64: 0, 2, 3, 1>}> : (tensor<3x3x7x7xf32>) -> tensor<3x7x7x3xf32>
    %6 = "llh.conv"(%4, %5) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NHWC>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<200x96x96x3xf32>, tensor<3x7x7x3xf32>) -> tensor<200x92x92x3xf32>
    %7 = "llh.transpose"(%6) <{perms = array<i64: 0, 3, 1, 2>}> : (tensor<200x92x92x3xf32>) -> tensor<200x3x92x92xf32>
    "llh.encoding_bind"(%7) <{encoding = #llh.encoding<shapes = @c200, @c3, @c92, @c92>}> : (tensor<200x3x92x92xf32>) -> ()
    "llh.encoding_bind"(%0) <{encoding = #llh.encoding<shapes = @c200, @c100, @c100, @c3>}> : (tensor<200x100x100x3xf32>) -> ()
    "llh.encoding_bind"(%1) <{encoding = #llh.encoding<shapes = @c3, @c7, @c7, @c3>}> : (tensor<3x7x7x3xf32>) -> ()
    "llh.encoding_bind"(%2) <{encoding = #llh.encoding<shapes = @c200, @c96, @c96, @c3>}> : (tensor<200x96x96x3xf32>) -> ()
    "llh.encoding_bind"(%4) <{encoding = #llh.encoding<shapes = @c200, @c96, @c96, @c3>}> : (tensor<200x96x96x3xf32>) -> ()
    "llh.encoding_bind"(%5) <{encoding = #llh.encoding<shapes = @c3, @c7, @c7, @c3>}> : (tensor<3x7x7x3xf32>) -> ()
    "llh.encoding_bind"(%6) <{encoding = #llh.encoding<shapes = @c200, @c92, @c92, @c3>}> : (tensor<200x92x92x3xf32>) -> ()
    "llh.encoding_bind"(%8) <{encoding = #llh.encoding<shapes = @c200, @c92, @c92, @c3>}> : (tensor<200x92x92x3xf32>) -> ()
    "llh.encoding_bind"(%9) <{encoding = #llh.encoding<shapes = @c3, @c7, @c7, @c3>}> : (tensor<3x7x7x3xf32>) -> ()
    "llh.encoding_bind"(%10) <{encoding = #llh.encoding<shapes = @c200, @c88, @c88, @c3>}> : (tensor<200x88x88x3xf32>) -> ()
    %8 = "llh.transpose"(%7) <{perms = array<i64: 0, 2, 3, 1>}> : (tensor<200x3x92x92xf32>) -> tensor<200x92x92x3xf32>
    %9 = "llh.transpose"(%arg0) <{perms = array<i64: 0, 2, 3, 1>}> : (tensor<3x3x7x7xf32>) -> tensor<3x7x7x3xf32>
    %10 = "llh.conv"(%8, %9) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NHWC>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<200x92x92x3xf32>, tensor<3x7x7x3xf32>) -> tensor<200x88x88x3xf32>
    %11 = "llh.transpose"(%10) <{perms = array<i64: 0, 3, 1, 2>}> : (tensor<200x88x88x3xf32>) -> tensor<200x3x88x88xf32>
    "llh.encoding_bind"(%11) <{encoding = #llh.encoding<shapes = @c200, @c3, @c88, @c88>}> : (tensor<200x3x88x88xf32>) -> ()
    "func.return"(%11, %arg0, %arg1, %3, %7) : (tensor<200x3x88x88xf32>, tensor<3x3x7x7xf32>, tensor<200x3x100x100xf32>, tensor<200x3x96x96xf32>, tensor<200x3x92x92xf32>) -> ()
  }) {entrance} : () -> ()
  "builtin.module"() <{sym_name = "__symbol__"}> ({
  ^bb0:
  }) : () -> ()
}) {builtin.gloabal_layout = "NCHW"} : () -> ()