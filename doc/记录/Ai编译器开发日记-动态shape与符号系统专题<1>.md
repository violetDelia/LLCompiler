## 动态shape

## 前言

在大多数推理场景中，模型的输入输出是固定的，但是在训练以及部分分布式推理的场景下，模型的输入的动态变化的。如果编译器的输入只支持固定的输入输出，例如这样：

```
func.func@main(%arg0:tensor<3x3x224x224xf32>) -> tensor<1x1000xf32> attributes {entrance}

```

上图是模型一个输入是（3，3，224，224）的模型的入口函数，将其编译之后，只能够计算输入是（3，3，224，224）的情况，如果需要计算输入大小为（2，3，512，512）的情况，需要重新编译一次模型，因此它只能够在推理的场景下运行。现在训练经常会出现不同大小的输入，尤其是大模型的训练下。因此需要在图层面动态的表达tensor的shape信息。

```

func.func@main(%arg0:tensor<?x3x?x?xf32>) -> tensor<?x1000xf32> attributes {entrance}
```

在MLIR中，会用？表示Tensor的shape信息是未知的，这样编译出来的模型就可以在动态变化的Tensor输入情况下持续运行。

但是，当用？去表达模型计算中的张量信息时，IR图中Tensor的shape信息严重缺失，没办法做图优化，以及一些后端优化。因此需要符号系统来表达shape之间的关系。

特别备注：因为IR的Op是笔者自己设计的，和xla的IR有些许的不同，xla的动态方案我就不再科普了

## torch-mlir 中的符号表达以及接入

在torch框架中，fx_graph 会有tensor计算中的符号信息，torch-mlir定义了一个特殊的Op来表示tensor的符号信息，笔者为了对接torch框架，也定义了和torch一样的Op，就一并介绍：

下图是解析fx_graph获得的resnet18IR图：

```
#map = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>
#map1 = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>
#map2 = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>
#map3 = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>
#map4 = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>
#map5 = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>
#map6 = affine_map<()[s0, s1] -> (s0, 512, 1, 1)>
#map7 = affine_map<()[s0, s1] -> (s0, 512)>
#map8 = affine_map<()[s0, s1] -> (s0, 1000)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<?x3x?x?xf32, {"0" = "s0", "1" = "c3", "2" = "s2", "3" = "s2"}>) -> tensor<?x1000xf32> attributes {entrance} {
    %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-19T22:36:12.138957+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>
    '''''' 
    %122 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-19T22:36:12.138957+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<1xi64>
    %123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
    %124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64
    "llh.symbolic_bind"(%arg0, %123, %124) <{expressions = #map}> : (tensor<?x3x?x?xf32, {"0" = "s0", "1" = "c3", "2" = "s2", "3" = "s2"}>, i64, i64) -> ()
    %125 = "llh.conv"(%arg0, %1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32, {"0" = "s0", "1" = "c3", "2" = "s2", "3" = "s2"}>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%125, %123, %124) <{expressions = #map1}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %126 = "llh.batch_norm"(%125, %2, %3, %63, %64) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%126, %123, %124) <{expressions = #map1}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %127 = "llh.relu"(%126) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%127, %123, %124) <{expressions = #map1}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %128 = "llh.max_pool"(%127) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%128, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %129 = "llh.conv"(%128, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%129, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %130 = "llh.batch_norm"(%129, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%130, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %131 = "llh.relu"(%130) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%131, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %132 = "llh.conv"(%131, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%132, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %133 = "llh.batch_norm"(%132, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%133, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %134 = "llh.add"(%133, %128) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%134, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %135 = "llh.relu"(%134) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%135, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %136 = "llh.conv"(%135, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%136, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %137 = "llh.batch_norm"(%136, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%137, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %138 = "llh.relu"(%137) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%138, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %139 = "llh.conv"(%138, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%139, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %140 = "llh.batch_norm"(%139, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%140, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %141 = "llh.add"(%140, %135) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%141, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %142 = "llh.relu"(%141) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%142, %123, %124) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %143 = "llh.conv"(%142, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%143, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %144 = "llh.batch_norm"(%143, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%144, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %145 = "llh.relu"(%144) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%145, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %146 = "llh.conv"(%145, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%146, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%147, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %148 = "llh.conv"(%142, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%148, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %149 = "llh.batch_norm"(%148, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%149, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %150 = "llh.add"(%147, %149) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%150, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %151 = "llh.relu"(%150) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%151, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %152 = "llh.conv"(%151, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%152, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %153 = "llh.batch_norm"(%152, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%153, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %154 = "llh.relu"(%153) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%154, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %155 = "llh.conv"(%154, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%155, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %156 = "llh.batch_norm"(%155, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%156, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %157 = "llh.add"(%156, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%157, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %158 = "llh.relu"(%157) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%158, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %159 = "llh.conv"(%158, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%159, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %160 = "llh.batch_norm"(%159, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%160, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %161 = "llh.relu"(%160) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%161, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %162 = "llh.conv"(%161, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%162, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %163 = "llh.batch_norm"(%162, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%163, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %164 = "llh.conv"(%158, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%164, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %165 = "llh.batch_norm"(%164, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%165, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %166 = "llh.add"(%163, %165) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%166, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %167 = "llh.relu"(%166) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%167, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %168 = "llh.conv"(%167, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%168, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %169 = "llh.batch_norm"(%168, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%169, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %170 = "llh.relu"(%169) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%170, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %171 = "llh.conv"(%170, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%171, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %172 = "llh.batch_norm"(%171, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%172, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %173 = "llh.add"(%172, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%173, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %174 = "llh.relu"(%173) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%174, %123, %124) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %175 = "llh.conv"(%174, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%175, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %176 = "llh.batch_norm"(%175, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%176, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %177 = "llh.relu"(%176) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%177, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %178 = "llh.conv"(%177, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%178, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %179 = "llh.batch_norm"(%178, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%179, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %180 = "llh.conv"(%174, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%180, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %181 = "llh.batch_norm"(%180, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%181, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %182 = "llh.add"(%179, %181) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%182, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %183 = "llh.relu"(%182) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%183, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %184 = "llh.conv"(%183, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%184, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %185 = "llh.batch_norm"(%184, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%185, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %186 = "llh.relu"(%185) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%186, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %187 = "llh.conv"(%186, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%187, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %188 = "llh.batch_norm"(%187, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%188, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %189 = "llh.add"(%188, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%189, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %190 = "llh.relu"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%190, %123, %124) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %191 = "llh.adaptive_average_pool"(%190) <{out_size = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>
    "llh.symbolic_bind"(%191, %123) <{expressions = #map6}> : (tensor<?x512x1x1xf32>, i64) -> ()
    %192 = "llh.flatten"(%191, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>
    "llh.symbolic_bind"(%192, %123) <{expressions = #map7}> : (tensor<?x512xf32>, i64) -> ()
    %193 = "llh.transpose"(%61) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>
    %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>
    %195 = "llh.add"(%194, %62) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>
    "llh.symbolic_bind"(%195, %123) <{expressions = #map8}> : (tensor<?x1000xf32>, i64) -> ()
    return %195 : tensor<?x1000xf32>
  }
}
```

torch-mlir 用torch.symbolic_int 在表示动态的符号, 它的表达和llh.torch_symbolic_int是一样的。

```
%123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
```

以上表示定义了符号s0来表示一个动态的int值用来表达张量的shape信息。
之后又定义了一个torch.symbolic_bind的Op来将符号信息和算子输出的tensor信息绑定。

```
#map3 = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>
%123 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
%124 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64
%147 = "llh.batch_norm"(%146, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
"llh.symbolic_bind"(%147, %123, %124) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
```

上图IR表示batch_norm的输出与s0和s2有关,其中expressions表达了shape和符号之间的关系，输出的第一个维度是s0，第二个维度是128，第三个维度是表达式(s1 - 1) floordiv 8 + 1的结果。

但是这种方案只能适用于对接torch，如果其他前端无法提供类似torch符号表达的功能，那就没办法去做符号相关的优化，因此笔者实现了另一种符号推导的方案。

## 符号推导

笔者定义了一个符号推导的接口，用来进行算子的符号和shape的推导：

```
def SymbolicInferShapeOpInterface: OpInterface<"SymbolicInferShapeOpInterface"> {
  let description = [{
    符号shepa推导的接口.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    /*retTy=*/"::llvm::LogicalResult",/*methodName=*/ "inferSymbolicShape">
  ];
}
```

然后为每一个Op实现了符号推导的方法；

以一个max_pool为例：

```
func.func @max_pool(%arg0: tensor<?x64x?x?xf32>) -> () attributes {entrance} {
 %129 = "llh.max_pool"(%arg0) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<*xf32>
  return 
}

```

经过符号推导之后：

```
module {
  "llh.symbolic_int"() <{sym_name = "s4"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @max_pool(%arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) attributes {entrance} {
    %0 = "llh.max_pool"(%arg0) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout `<NCHW>`, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s3, @s4>>
    return
  }
  module @__symbol__ {
  }
}
```

## 符号优化

当我们知道符号之间的关系后，可以对符号进行优化，比如：

```
"llh.symbolic_int"() <{sym_name = "s5"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s4"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
func.func @relation_eq(%arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, %arg1: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) ->(i64) attributes {entrance} {
  %2 = "llh.constant"() <{symbol = @c0, value = 2 : i64}> : () -> i64
  %0 = "llh.add"(%arg0, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
  %1 = "llh.add"(%arg1, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>
  "llh.encoding_bind"(%1) <{encoding = #llh.encoding<shapes = @s3, @c64, @s4, @s5>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> ()
  %193 = "llh.dim"(%1, %2) <{symbol = @s4}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>, i64) -> i64
  return %193: i64
}

module @__symbol__ {
  "llh.symbol_relation"() <{relation = @s5, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s2}> : () -> ()
  "llh.symbol_relation"() <{relation = @s4, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s1}> : () -> ()
  "llh.symbol_relation"() <{relation = @s3, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s0}> : () -> ()
  "llh.symbol_relation"() <{relation = @s3, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s0}> : () -> ()
}
```

在 __symbol__ 里面保存这记录下来的符号关系，例如这条

`"llh.symbol_relation"() <{relation = @s4, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s1}> : () -> ()`

表示符号s4和s1是相等的，就会对全图的符号进行优化，将s4替换为s1，优化之后的结果如下：

```
module {
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  func.func @relation_eq(%arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, %arg1: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> i64 attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c0, value = 2 : i64}> : () -> i64
    %1 = "llh.add"(%arg0, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
    %2 = "llh.add"(%arg1, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
    "llh.encoding_bind"(%2) <{encoding = #llh.encoding<shapes = @s0, @c64, @s1, @s2>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> ()
    %3 = "llh.dim"(%2, %0) <{symbol = @s1}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, i64) -> i64
    return %3 : i64
  }
  module @__symbol__ {
  }
}
```

## 推导示例

emm这套符号系统笔者前前后后花了很久的时间才设计好，之所以如此大费周章是因为它在后端优化中依然会起到至关重要的作用，现在演示的只有表达shape信息的功能，之后会介绍这套符号系统在其他方面的作用。

以下是resnet推导的IR图示例：

推导之前：

```
func.func @main(%arg0: tensor<?x3x?x?xf32>) -> tensor<?x1000xf32> attributes {entrance}
''''''
%189 = "llh.batch_norm"(%188, %62, %63, %123, %124) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    %190 = "llh.add"(%189, %184) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    %191 = "llh.relu"(%190) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    %192 = "llh.adaptive_average_pool"(%191) <{out_size = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>
    %193 = "llh.dim"(%192, %2) : (tensor<?x512x1x1xf32>, i64) -> i64
    %194 = "llh.dim"(%192, %3) : (tensor<?x512x1x1xf32>, i64) -> i64
    %195 = "llh.dim"(%192, %1) : (tensor<?x512x1x1xf32>, i64) -> i64
    %196 = "llh.dim"(%192, %0) : (tensor<?x512x1x1xf32>, i64) -> i64
    %197 = "llh.mul"(%194, %195) : (i64, i64) -> i64
    %198 = "llh.mul"(%197, %196) : (i64, i64) -> i64
    %199 = "llh.reshape"(%192, %193, %198) : (tensor<?x512x1x1xf32>, i64, i64) -> tensor<?x512xf32>
    %200 = "llh.transpose"(%64) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>
    %201 = "llh.matmul"(%199, %200) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>
    %202 = "llh.add"(%201, %65) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>
    return %202 : tensor<?x1000xf32>
```

推导之后：

可以看到输出的s0正是输入的s0，说明推导正确。

```

module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "s43"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s42"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s41"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s40"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s35"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s34"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s33"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s32"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s31"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s30"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s25"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s24"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s23"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s22"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s21"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s20"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s15"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s14"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s13"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s12"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s11"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s10"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s7"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s6"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1000"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c256"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c128"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c7"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  "llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
  func.func @main(%arg0: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>) -> tensor<?x1000xf32, #llh.encoding<shapes = @s0, @c1000>> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-19T23:47:39.318672+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>
    ''''''
    %125 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-10-19T23:47:39.318672+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<1xi64, #llh.encoding<shapes = @c1>>
    %126 = "llh.conv"(%arg0, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>
    %127 = "llh.batch_norm"(%126, %5, %6, %66, %67) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>
    %128 = "llh.relu"(%127) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>
    %129 = "llh.max_pool"(%128) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %130 = "llh.conv"(%129, %7) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s6, @s7>>
    %131 = "llh.batch_norm"(%130, %8, %9, %69, %70) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s6, @s7>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s6, @s7>>
    %132 = "llh.relu"(%131) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s6, @s7>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s6, @s7>>
    %133 = "llh.conv"(%132, %10) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s6, @s7>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %134 = "llh.batch_norm"(%133, %11, %12, %72, %73) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %135 = "llh.add"(%134, %129) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %136 = "llh.relu"(%135) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %137 = "llh.conv"(%136, %13) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s10, @s11>>
    %138 = "llh.batch_norm"(%137, %14, %15, %75, %76) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s10, @s11>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s10, @s11>>
    %139 = "llh.relu"(%138) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s10, @s11>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s10, @s11>>
    %140 = "llh.conv"(%139, %16) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s10, @s11>>, tensor<64x64x3x3xf32, #llh.encoding<shapes = @c64, @c64, @c3, @c3>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %141 = "llh.batch_norm"(%140, %17, %18, %78, %79) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %142 = "llh.add"(%141, %136) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %143 = "llh.relu"(%142) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>
    %144 = "llh.conv"(%143, %19) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>, tensor<128x64x3x3xf32, #llh.encoding<shapes = @c128, @c64, @c3, @c3>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s14, @s15>>
    %145 = "llh.batch_norm"(%144, %20, %21, %81, %82) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s14, @s15>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s14, @s15>>
    %146 = "llh.relu"(%145) : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s14, @s15>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s14, @s15>>
    %147 = "llh.conv"(%146, %22) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s14, @s15>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %148 = "llh.batch_norm"(%147, %23, %24, %84, %85) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %149 = "llh.conv"(%143, %25) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s12, @s13>>, tensor<128x64x1x1xf32, #llh.encoding<shapes = @c128, @c64, @c1, @c1>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %150 = "llh.batch_norm"(%149, %26, %27, %87, %88) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %151 = "llh.add"(%148, %150) : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>, tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %152 = "llh.relu"(%151) : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %153 = "llh.conv"(%152, %28) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s20, @s21>>
    %154 = "llh.batch_norm"(%153, %29, %30, %90, %91) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s20, @s21>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s20, @s21>>
    %155 = "llh.relu"(%154) : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s20, @s21>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s20, @s21>>
    %156 = "llh.conv"(%155, %31) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s20, @s21>>, tensor<128x128x3x3xf32, #llh.encoding<shapes = @c128, @c128, @c3, @c3>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %157 = "llh.batch_norm"(%156, %32, %33, %93, %94) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>, tensor<128xf32, #llh.encoding<shapes = @c128>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %158 = "llh.add"(%157, %152) : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>, tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %159 = "llh.relu"(%158) : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>) -> tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>
    %160 = "llh.conv"(%159, %34) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>, tensor<256x128x3x3xf32, #llh.encoding<shapes = @c256, @c128, @c3, @c3>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s24, @s25>>
    %161 = "llh.batch_norm"(%160, %35, %36, %96, %97) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s24, @s25>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s24, @s25>>
    %162 = "llh.relu"(%161) : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s24, @s25>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s24, @s25>>
    %163 = "llh.conv"(%162, %37) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s24, @s25>>, tensor<256x256x3x3xf32, #llh.encoding<shapes = @c256, @c256, @c3, @c3>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %164 = "llh.batch_norm"(%163, %38, %39, %99, %100) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %165 = "llh.conv"(%159, %40) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32, #llh.encoding<shapes = @s0, @c128, @s22, @s23>>, tensor<256x128x1x1xf32, #llh.encoding<shapes = @c256, @c128, @c1, @c1>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %166 = "llh.batch_norm"(%165, %41, %42, %102, %103) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %167 = "llh.add"(%164, %166) : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>, tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %168 = "llh.relu"(%167) : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %169 = "llh.conv"(%168, %43) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>, tensor<256x256x3x3xf32, #llh.encoding<shapes = @c256, @c256, @c3, @c3>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s30, @s31>>
    %170 = "llh.batch_norm"(%169, %44, %45, %105, %106) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s30, @s31>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s30, @s31>>
    %171 = "llh.relu"(%170) : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s30, @s31>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s30, @s31>>
    %172 = "llh.conv"(%171, %46) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s30, @s31>>, tensor<256x256x3x3xf32, #llh.encoding<shapes = @c256, @c256, @c3, @c3>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %173 = "llh.batch_norm"(%172, %47, %48, %108, %109) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>, tensor<256xf32, #llh.encoding<shapes = @c256>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %174 = "llh.add"(%173, %168) : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>, tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %175 = "llh.relu"(%174) : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>) -> tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>
    %176 = "llh.conv"(%175, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>, tensor<512x256x3x3xf32, #llh.encoding<shapes = @c512, @c256, @c3, @c3>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s34, @s35>>
    %177 = "llh.batch_norm"(%176, %50, %51, %111, %112) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s34, @s35>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s34, @s35>>
    %178 = "llh.relu"(%177) : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s34, @s35>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s34, @s35>>
    %179 = "llh.conv"(%178, %52) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s34, @s35>>, tensor<512x512x3x3xf32, #llh.encoding<shapes = @c512, @c512, @c3, @c3>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %180 = "llh.batch_norm"(%179, %53, %54, %114, %115) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %181 = "llh.conv"(%175, %55) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, layout = #llh.Layout<NCHW>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32, #llh.encoding<shapes = @s0, @c256, @s32, @s33>>, tensor<512x256x1x1xf32, #llh.encoding<shapes = @c512, @c256, @c1, @c1>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %182 = "llh.batch_norm"(%181, %56, %57, %117, %118) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %183 = "llh.add"(%180, %182) : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>, tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %184 = "llh.relu"(%183) : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %185 = "llh.conv"(%184, %58) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>, tensor<512x512x3x3xf32, #llh.encoding<shapes = @c512, @c512, @c3, @c3>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s40, @s41>>
    %186 = "llh.batch_norm"(%185, %59, %60, %120, %121) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s40, @s41>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s40, @s41>>
    %187 = "llh.relu"(%186) : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s40, @s41>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s40, @s41>>
    %188 = "llh.conv"(%187, %61) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s40, @s41>>, tensor<512x512x3x3xf32, #llh.encoding<shapes = @c512, @c512, @c3, @c3>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %189 = "llh.batch_norm"(%188, %62, %63, %123, %124) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>, tensor<512xf32, #llh.encoding<shapes = @c512>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %190 = "llh.add"(%189, %184) : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>, tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %191 = "llh.relu"(%190) : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
    %192 = "llh.adaptive_average_pool"(%191) <{out_size = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) -> tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>
    %193 = "llh.dim"(%192, %2) <{symbol = @s0}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %194 = "llh.dim"(%192, %3) <{symbol = @c512}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %195 = "llh.dim"(%192, %1) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %196 = "llh.dim"(%192, %0) <{symbol = @c1}> : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
    %197 = "llh.mul"(%194, %195) <{symbol = @c512}> : (i64, i64) -> i64
    %198 = "llh.mul"(%197, %196) <{symbol = @c512}> : (i64, i64) -> i64
    %199 = "llh.reshape"(%192, %193, %198) : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64, i64) -> tensor<?x512xf32, #llh.encoding<shapes = @s0, @c512>>
    %200 = "llh.transpose"(%64) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32, #llh.encoding<shapes = @c1000, @c512>>) -> tensor<512x1000xf32, #llh.encoding<shapes = @c512, @c1000>>
    %201 = "llh.matmul"(%199, %200) : (tensor<?x512xf32, #llh.encoding<shapes = @s0, @c512>>, tensor<512x1000xf32, #llh.encoding<shapes = @c512, @c1000>>) -> tensor<?x1000xf32, #llh.encoding<shapes = @s0, @c1000>>
    %202 = "llh.add"(%201, %65) : (tensor<?x1000xf32, #llh.encoding<shapes = @s0, @c1000>>, tensor<1000xf32, #llh.encoding<shapes = @c1000>>) -> tensor<?x1000xf32, #llh.encoding<shapes = @s0, @c1000>>
    return %202 : tensor<?x1000xf32, #llh.encoding<shapes = @s0, @c1000>>
  }
  module @__symbol__ {
    "llh.symbol_binary_relation"() <{relation_kind = #llh.SymbolRelation<Mul>, relations_lhs = @c1, relations_rhs = @c512, symbol = @c512}> : () -> ()
    "llh.symbol_binary_relation"() <{relation_kind = #llh.SymbolRelation<Mul>, relations_lhs = @c512, relations_rhs = @c1, symbol = @c512}> : () -> ()
    "llh.symbol_binary_relation"() <{relation_kind = #llh.SymbolRelation<Mul>, relations_lhs = @c1, relations_rhs = @c512, symbol = @c512}> : () -> ()
    "llh.symbol_binary_relation"() <{relation_kind = #llh.SymbolRelation<Mul>, relations_lhs = @c512, relations_rhs = @c1, symbol = @c512}> : () -> ()
  }
}
```
