// -----// IR Dump Before Operationlegalization (operation-legalization) //----- //
#map = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>
#map1 = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 2 + 1, (s1 - 1) floordiv 2 + 1)>
#map2 = affine_map<()[s0, s1] -> (s0, 64, (s1 - 1) floordiv 4 + 1, (s1 - 1) floordiv 4 + 1)>
#map3 = affine_map<()[s0, s1] -> (s0, 128, (s1 - 1) floordiv 8 + 1, (s1 - 1) floordiv 8 + 1)>
#map4 = affine_map<()[s0, s1] -> (s0, 256, (s1 - 1) floordiv 16 + 1, (s1 - 1) floordiv 16 + 1)>
#map5 = affine_map<()[s0, s1] -> (s0, 512, (s1 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>
#map6 = affine_map<()[s0, s1] -> (s0, 512, 1, 1)>
#map7 = affine_map<()[s0, s1] -> (s0, 512)>
#map8 = affine_map<()[s0, s1] -> (s0, 1000)>
"builtin.module"() ({
  "func.func"() <{function_type = (i64, i64, tensor<?x3x?x?xf32>) -> tensor<?x1000xf32>, sym_name = "main"}> ({
  ^bb0(%arg0: i64, %arg1: i64, %arg2: tensor<?x3x?x?xf32>):
    %0 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>
    %1 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.weight.npy"}> : () -> tensor<64xf32>
    %2 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.bias.npy"}> : () -> tensor<64xf32>
    %3 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>
    %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.weight.npy"}> : () -> tensor<64xf32>
    %5 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.bias.npy"}> : () -> tensor<64xf32>
    %6 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>
    %7 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.weight.npy"}> : () -> tensor<64xf32>
    %8 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.bias.npy"}> : () -> tensor<64xf32>
    %9 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv1.weight.npy"}> : () -> tensor<64x64x3x3xf32>
    %10 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.weight.npy"}> : () -> tensor<64xf32>
    %11 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.bias.npy"}> : () -> tensor<64xf32>
    %12 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___conv2.weight.npy"}> : () -> tensor<64x64x3x3xf32>
    %13 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.weight.npy"}> : () -> tensor<64xf32>
    %14 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.bias.npy"}> : () -> tensor<64xf32>
    %15 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv1.weight.npy"}> : () -> tensor<128x64x3x3xf32>
    %16 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.weight.npy"}> : () -> tensor<128xf32>
    %17 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.bias.npy"}> : () -> tensor<128xf32>
    %18 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>
    %19 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.weight.npy"}> : () -> tensor<128xf32>
    %20 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.bias.npy"}> : () -> tensor<128xf32>
    %21 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"}> : () -> tensor<128x64x1x1xf32>
    %22 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"}> : () -> tensor<128xf32>
    %23 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"}> : () -> tensor<128xf32>
    %24 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv1.weight.npy"}> : () -> tensor<128x128x3x3xf32>
    %25 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.weight.npy"}> : () -> tensor<128xf32>
    %26 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.bias.npy"}> : () -> tensor<128xf32>
    %27 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___conv2.weight.npy"}> : () -> tensor<128x128x3x3xf32>
    %28 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.weight.npy"}> : () -> tensor<128xf32>
    %29 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.bias.npy"}> : () -> tensor<128xf32>
    %30 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv1.weight.npy"}> : () -> tensor<256x128x3x3xf32>
    %31 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.weight.npy"}> : () -> tensor<256xf32>
    %32 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.bias.npy"}> : () -> tensor<256xf32>
    %33 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>
    %34 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.weight.npy"}> : () -> tensor<256xf32>
    %35 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.bias.npy"}> : () -> tensor<256xf32>
    %36 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"}> : () -> tensor<256x128x1x1xf32>
    %37 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"}> : () -> tensor<256xf32>
    %38 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"}> : () -> tensor<256xf32>
    %39 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv1.weight.npy"}> : () -> tensor<256x256x3x3xf32>
    %40 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.weight.npy"}> : () -> tensor<256xf32>
    %41 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.bias.npy"}> : () -> tensor<256xf32>
    %42 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___conv2.weight.npy"}> : () -> tensor<256x256x3x3xf32>
    %43 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.weight.npy"}> : () -> tensor<256xf32>
    %44 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.bias.npy"}> : () -> tensor<256xf32>
    %45 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv1.weight.npy"}> : () -> tensor<512x256x3x3xf32>
    %46 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.weight.npy"}> : () -> tensor<512xf32>
    %47 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.bias.npy"}> : () -> tensor<512xf32>
    %48 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>
    %49 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.weight.npy"}> : () -> tensor<512xf32>
    %50 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.bias.npy"}> : () -> tensor<512xf32>
    %51 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"}> : () -> tensor<512x256x1x1xf32>
    %52 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"}> : () -> tensor<512xf32>
    %53 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"}> : () -> tensor<512xf32>
    %54 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv1.weight.npy"}> : () -> tensor<512x512x3x3xf32>
    %55 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.weight.npy"}> : () -> tensor<512xf32>
    %56 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.bias.npy"}> : () -> tensor<512xf32>
    %57 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___conv2.weight.npy"}> : () -> tensor<512x512x3x3xf32>
    %58 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.weight.npy"}> : () -> tensor<512xf32>
    %59 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.bias.npy"}> : () -> tensor<512xf32>
    %60 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.weight.npy"}> : () -> tensor<1000x512xf32>
    %61 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___fc.bias.npy"}> : () -> tensor<1000xf32>
    %62 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_mean.npy"}> : () -> tensor<64xf32>
    %63 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.running_var.npy"}> : () -> tensor<64xf32>
    %64 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/L__self___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %65 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"}> : () -> tensor<64xf32>
    %66 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"}> : () -> tensor<64xf32>
    %67 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %68 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"}> : () -> tensor<64xf32>
    %69 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"}> : () -> tensor<64xf32>
    %70 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
    %71 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"}> : () -> tensor<64xf32>
    %72 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"}> : () -> tensor<64xf32>
    %73 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %74 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"}> : () -> tensor<64xf32>
    %75 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"}> : () -> tensor<64xf32>
    %76 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
    %77 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"}> : () -> tensor<128xf32>
    %78 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"}> : () -> tensor<128xf32>
    %79 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %80 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"}> : () -> tensor<128xf32>
    %81 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"}> : () -> tensor<128xf32>
    %82 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
    %83 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"}> : () -> tensor<128xf32>
    %84 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"}> : () -> tensor<128xf32>
    %85 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %86 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"}> : () -> tensor<128xf32>
    %87 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"}> : () -> tensor<128xf32>
    %88 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %89 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"}> : () -> tensor<128xf32>
    %90 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"}> : () -> tensor<128xf32>
    %91 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
    %92 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"}> : () -> tensor<256xf32>
    %93 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"}> : () -> tensor<256xf32>
    %94 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %95 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"}> : () -> tensor<256xf32>
    %96 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"}> : () -> tensor<256xf32>
    %97 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
    %98 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"}> : () -> tensor<256xf32>
    %99 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"}> : () -> tensor<256xf32>
    %100 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %101 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"}> : () -> tensor<256xf32>
    %102 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"}> : () -> tensor<256xf32>
    %103 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %104 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"}> : () -> tensor<256xf32>
    %105 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"}> : () -> tensor<256xf32>
    %106 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
    %107 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"}> : () -> tensor<512xf32>
    %108 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"}> : () -> tensor<512xf32>
    %109 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %110 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"}> : () -> tensor<512xf32>
    %111 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"}> : () -> tensor<512xf32>
    %112 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
    %113 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"}> : () -> tensor<512xf32>
    %114 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"}> : () -> tensor<512xf32>
    %115 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %116 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"}> : () -> tensor<512xf32>
    %117 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"}> : () -> tensor<512xf32>
    %118 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"}> : () -> tensor<i64>
    %119 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"}> : () -> tensor<512xf32>
    %120 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"}> : () -> tensor<512xf32>
    %121 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-25T03:10:36.365688+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"}> : () -> tensor<i64>
    %122 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
    %123 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64
    "llh.symbolic_bind"(%arg2, %122, %123) <{expressions = #map}> : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %124 = "llh.conv"(%arg2, %0) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%124, %122, %123) <{expressions = #map1}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %125 = "llh.batch_norm"(%124, %1, %2, %62, %63) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%125, %122, %123) <{expressions = #map1}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %126 = "llh.relu"(%125) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%126, %122, %123) <{expressions = #map1}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %127 = "llh.max_pool"(%126) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%127, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %128 = "llh.conv"(%127, %3) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%128, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %129 = "llh.batch_norm"(%128, %4, %5, %65, %66) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%129, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %130 = "llh.relu"(%129) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%130, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %131 = "llh.conv"(%130, %6) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%131, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %132 = "llh.batch_norm"(%131, %7, %8, %68, %69) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%132, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %133 = "llh.add"(%132, %127) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%133, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %134 = "llh.relu"(%133) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%134, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %135 = "llh.conv"(%134, %9) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%135, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %136 = "llh.batch_norm"(%135, %10, %11, %71, %72) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%136, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %137 = "llh.relu"(%136) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%137, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %138 = "llh.conv"(%137, %12) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%138, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %139 = "llh.batch_norm"(%138, %13, %14, %74, %75) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%139, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %140 = "llh.add"(%139, %134) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%140, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %141 = "llh.relu"(%140) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%141, %122, %123) <{expressions = #map2}> : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %142 = "llh.conv"(%141, %15) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%142, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %143 = "llh.batch_norm"(%142, %16, %17, %77, %78) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%143, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %144 = "llh.relu"(%143) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%144, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %145 = "llh.conv"(%144, %18) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%145, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %146 = "llh.batch_norm"(%145, %19, %20, %80, %81) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%146, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %147 = "llh.conv"(%141, %21) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%147, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %148 = "llh.batch_norm"(%147, %22, %23, %83, %84) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%148, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %149 = "llh.add"(%146, %148) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%149, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %150 = "llh.relu"(%149) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%150, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %151 = "llh.conv"(%150, %24) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%151, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %152 = "llh.batch_norm"(%151, %25, %26, %86, %87) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%152, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %153 = "llh.relu"(%152) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%153, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %154 = "llh.conv"(%153, %27) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%154, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %155 = "llh.batch_norm"(%154, %28, %29, %89, %90) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%155, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %156 = "llh.add"(%155, %150) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%156, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %157 = "llh.relu"(%156) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%157, %122, %123) <{expressions = #map3}> : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %158 = "llh.conv"(%157, %30) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%158, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %159 = "llh.batch_norm"(%158, %31, %32, %92, %93) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%159, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %160 = "llh.relu"(%159) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%160, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %161 = "llh.conv"(%160, %33) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%161, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %162 = "llh.batch_norm"(%161, %34, %35, %95, %96) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%162, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %163 = "llh.conv"(%157, %36) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%163, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %164 = "llh.batch_norm"(%163, %37, %38, %98, %99) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%164, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %165 = "llh.add"(%162, %164) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%165, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %166 = "llh.relu"(%165) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%166, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %167 = "llh.conv"(%166, %39) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%167, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %168 = "llh.batch_norm"(%167, %40, %41, %101, %102) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%168, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %169 = "llh.relu"(%168) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%169, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %170 = "llh.conv"(%169, %42) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%170, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %171 = "llh.batch_norm"(%170, %43, %44, %104, %105) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%171, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %172 = "llh.add"(%171, %166) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%172, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %173 = "llh.relu"(%172) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%173, %122, %123) <{expressions = #map4}> : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %174 = "llh.conv"(%173, %45) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%174, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %175 = "llh.batch_norm"(%174, %46, %47, %107, %108) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%175, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %176 = "llh.relu"(%175) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%176, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %177 = "llh.conv"(%176, %48) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%177, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %178 = "llh.batch_norm"(%177, %49, %50, %110, %111) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%178, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %179 = "llh.conv"(%173, %51) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%179, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %180 = "llh.batch_norm"(%179, %52, %53, %113, %114) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%180, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %181 = "llh.add"(%178, %180) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%181, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %182 = "llh.relu"(%181) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%182, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %183 = "llh.conv"(%182, %54) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%183, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %184 = "llh.batch_norm"(%183, %55, %56, %116, %117) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%184, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %185 = "llh.relu"(%184) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%185, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %186 = "llh.conv"(%185, %57) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%186, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %187 = "llh.batch_norm"(%186, %58, %59, %119, %120) <{epsilon = 1.000000e-05 : f64, momentum = 1.000000e-01 : f64}> : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%187, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %188 = "llh.add"(%187, %182) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%188, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %189 = "llh.relu"(%188) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%189, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %190 = "llh.adaptive_average_pool"(%189) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>
    "llh.symbolic_bind"(%190, %122) <{expressions = #map6}> : (tensor<?x512x1x1xf32>, i64) -> ()
    %191 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %192 = "llh.flatten"(%190, %191) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>
    "llh.symbolic_bind"(%192, %122) <{expressions = #map7}> : (tensor<?x512xf32>, i64) -> ()
    %193 = "llh.transpose"(%60) <{perms = array<i64: 1, 0>}> : (tensor<1000x512xf32>) -> tensor<512x1000xf32>
    %194 = "llh.matmul"(%192, %193) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>
    %195 = "llh.add"(%194, %61) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>
    "llh.symbolic_bind"(%195, %122) <{expressions = #map8}> : (tensor<?x1000xf32>, i64) -> ()
    "func.return"(%195) : (tensor<?x1000xf32>) -> ()
  }) : () -> ()
}) {builtin.gloabal_layout = "NCHW"} : () -> ()


