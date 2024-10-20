func.func @add_float() ->(f32,f32,f32) attributes {entrance} {
  %0 = "llh.constant"() <{value = 0. : f32}> : () -> f32
  %1 = "llh.constant"() <{value = 1. : f32}> : () -> f32
  %2 = "llh.constant"() <{value = 2. : f32}> : () -> f32
  %3 = "llh.constant"() <{value = 3. : f32}> : () -> f32
  %add_0_l = "llh.add"(%0, %2) : (f32, f32) -> f32
  %add_0_r = "llh.add"(%2, %0) : (f32, f32) -> f32
  %103 = "llh.sub"(%2, %3) : (f32, f32) -> f32
  %add_sub_l = "llh.add"(%103, %3) : (f32, f32) -> f32
  %105 = "llh.add"(%3, %103) : (f32, f32) -> f32
  return %add_0_l,%add_0_r,%add_sub_l : f32,f32,f32
}