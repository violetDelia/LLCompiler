module {
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  func.func @dim_and_const(%arg0: tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>) attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %4 = "llh.dim"(%arg0, %2) <{symbol = @s0}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %5 = "llh.dim"(%arg0, %3) <{symbol = @c512}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %6 = "llh.dim"(%arg0, %1) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %7 = "llh.dim"(%arg0, %0) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    return
  }
  module @__symbol__ {
  }
}

// -----
module {
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @mul(%arg0: tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>) attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %4 = "llh.dim"(%arg0, %2) <{symbol = @s0}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %5 = "llh.dim"(%arg0, %3) <{symbol = @c512}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %6 = "llh.dim"(%arg0, %1) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %7 = "llh.dim"(%arg0, %0) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %8 = "llh.mul"(%5, %6) <{symbol = @s1}> : (i64, i64) -> i64
    %9 = "llh.mul"(%5, %6) <{symbol = @s2}> : (i64, i64) -> i64
    return
  }
  module @__symbol__ {
  }
}

