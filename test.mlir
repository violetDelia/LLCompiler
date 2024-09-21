builtin.module attributes  {"builtin.gloabal_layout" = "NCHW"} {
  func.func @main(%0 : i64, %1 : i64, %2 : tensor<?x3x?x?xf32>) -> tensor<?x1000xf32> {
    %3 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/L__self___conv1.weight.npy"} : () -> tensor<64x3x7x7xf32>
    %4 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/L__self___bn1.weight.npy"} : () -> tensor<64xf32>
    %5 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/L__self___bn1.bias.npy"} : () -> tensor<64xf32>
    %6 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___conv1.weight.npy"} : () -> tensor<64x64x3x3xf32>
    %7 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn1.weight.npy"} : () -> tensor<64xf32>
    %8 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn1.bias.npy"} : () -> tensor<64xf32>
    %9 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___conv2.weight.npy"} : () -> tensor<64x64x3x3xf32>
    %10 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn2.weight.npy"} : () -> tensor<64xf32>
    %11 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn2.bias.npy"} : () -> tensor<64xf32>
    %12 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___conv1.weight.npy"} : () -> tensor<64x64x3x3xf32>
    %13 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn1.weight.npy"} : () -> tensor<64xf32>
    %14 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn1.bias.npy"} : () -> tensor<64xf32>
    %15 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___conv2.weight.npy"} : () -> tensor<64x64x3x3xf32>
    %16 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn2.weight.npy"} : () -> tensor<64xf32>
    %17 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn2.bias.npy"} : () -> tensor<64xf32>
    %18 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___conv1.weight.npy"} : () -> tensor<128x64x3x3xf32>
    %19 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn1.weight.npy"} : () -> tensor<128xf32>
    %20 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn1.bias.npy"} : () -> tensor<128xf32>
    %21 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___conv2.weight.npy"} : () -> tensor<128x128x3x3xf32>
    %22 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn2.weight.npy"} : () -> tensor<128xf32>
    %23 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn2.bias.npy"} : () -> tensor<128xf32>
    %24 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___downsample_0.weight.npy"} : () -> tensor<128x64x1x1xf32>
    %25 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___downsample_1.weight.npy"} : () -> tensor<128xf32>
    %26 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___downsample_1.bias.npy"} : () -> tensor<128xf32>
    %27 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___conv1.weight.npy"} : () -> tensor<128x128x3x3xf32>
    %28 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn1.weight.npy"} : () -> tensor<128xf32>
    %29 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn1.bias.npy"} : () -> tensor<128xf32>
    %30 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___conv2.weight.npy"} : () -> tensor<128x128x3x3xf32>
    %31 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn2.weight.npy"} : () -> tensor<128xf32>
    %32 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn2.bias.npy"} : () -> tensor<128xf32>
    %33 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___conv1.weight.npy"} : () -> tensor<256x128x3x3xf32>
    %34 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn1.weight.npy"} : () -> tensor<256xf32>
    %35 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn1.bias.npy"} : () -> tensor<256xf32>
    %36 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___conv2.weight.npy"} : () -> tensor<256x256x3x3xf32>
    %37 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn2.weight.npy"} : () -> tensor<256xf32>
    %38 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn2.bias.npy"} : () -> tensor<256xf32>
    %39 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___downsample_0.weight.npy"} : () -> tensor<256x128x1x1xf32>
    %40 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___downsample_1.weight.npy"} : () -> tensor<256xf32>
    %41 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___downsample_1.bias.npy"} : () -> tensor<256xf32>
    %42 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___conv1.weight.npy"} : () -> tensor<256x256x3x3xf32>
    %43 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn1.weight.npy"} : () -> tensor<256xf32>
    %44 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn1.bias.npy"} : () -> tensor<256xf32>
    %45 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___conv2.weight.npy"} : () -> tensor<256x256x3x3xf32>
    %46 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn2.weight.npy"} : () -> tensor<256xf32>
    %47 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn2.bias.npy"} : () -> tensor<256xf32>
    %48 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___conv1.weight.npy"} : () -> tensor<512x256x3x3xf32>
    %49 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn1.weight.npy"} : () -> tensor<512xf32>
    %50 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn1.bias.npy"} : () -> tensor<512xf32>
    %51 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___conv2.weight.npy"} : () -> tensor<512x512x3x3xf32>
    %52 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn2.weight.npy"} : () -> tensor<512xf32>
    %53 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn2.bias.npy"} : () -> tensor<512xf32>
    %54 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___downsample_0.weight.npy"} : () -> tensor<512x256x1x1xf32>
    %55 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___downsample_1.weight.npy"} : () -> tensor<512xf32>
    %56 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___downsample_1.bias.npy"} : () -> tensor<512xf32>
    %57 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___conv1.weight.npy"} : () -> tensor<512x512x3x3xf32>
    %58 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn1.weight.npy"} : () -> tensor<512xf32>
    %59 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn1.bias.npy"} : () -> tensor<512xf32>
    %60 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___conv2.weight.npy"} : () -> tensor<512x512x3x3xf32>
    %61 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn2.weight.npy"} : () -> tensor<512xf32>
    %62 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn2.bias.npy"} : () -> tensor<512xf32>
    %63 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/L__self___fc.weight.npy"} : () -> tensor<1000x512xf32>
    %64 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/L__self___fc.bias.npy"} : () -> tensor<1000xf32>
    %65 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/L__self___bn1.running_mean.npy"} : () -> tensor<64xf32>
    %66 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/L__self___bn1.running_var.npy"} : () -> tensor<64xf32>
    %67 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/L__self___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %68 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn1.running_mean.npy"} : () -> tensor<64xf32>
    %69 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn1.running_var.npy"} : () -> tensor<64xf32>
    %70 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %71 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn2.running_mean.npy"} : () -> tensor<64xf32>
    %72 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn2.running_var.npy"} : () -> tensor<64xf32>
    %73 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___0___bn2.num_batches_tracked.npy"} : () -> tensor<i64>
    %74 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn1.running_mean.npy"} : () -> tensor<64xf32>
    %75 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn1.running_var.npy"} : () -> tensor<64xf32>
    %76 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %77 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn2.running_mean.npy"} : () -> tensor<64xf32>
    %78 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn2.running_var.npy"} : () -> tensor<64xf32>
    %79 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer1___1___bn2.num_batches_tracked.npy"} : () -> tensor<i64>
    %80 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn1.running_mean.npy"} : () -> tensor<128xf32>
    %81 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn1.running_var.npy"} : () -> tensor<128xf32>
    %82 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %83 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn2.running_mean.npy"} : () -> tensor<128xf32>
    %84 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn2.running_var.npy"} : () -> tensor<128xf32>
    %85 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___bn2.num_batches_tracked.npy"} : () -> tensor<i64>
    %86 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___downsample_1.running_mean.npy"} : () -> tensor<128xf32>
    %87 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___downsample_1.running_var.npy"} : () -> tensor<128xf32>
    %88 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___0___downsample_1.num_batches_tracked.npy"} : () -> tensor<i64>
    %89 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn1.running_mean.npy"} : () -> tensor<128xf32>
    %90 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn1.running_var.npy"} : () -> tensor<128xf32>
    %91 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %92 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn2.running_mean.npy"} : () -> tensor<128xf32>
    %93 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn2.running_var.npy"} : () -> tensor<128xf32>
    %94 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer2___1___bn2.num_batches_tracked.npy"} : () -> tensor<i64>
    %95 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn1.running_mean.npy"} : () -> tensor<256xf32>
    %96 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn1.running_var.npy"} : () -> tensor<256xf32>
    %97 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %98 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn2.running_mean.npy"} : () -> tensor<256xf32>
    %99 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn2.running_var.npy"} : () -> tensor<256xf32>
    %100 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___bn2.num_batches_tracked.npy"} : () -> tensor<i64>
    %101 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___downsample_1.running_mean.npy"} : () -> tensor<256xf32>
    %102 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___downsample_1.running_var.npy"} : () -> tensor<256xf32>
    %103 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___0___downsample_1.num_batches_tracked.npy"} : () -> tensor<i64>
    %104 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn1.running_mean.npy"} : () -> tensor<256xf32>
    %105 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn1.running_var.npy"} : () -> tensor<256xf32>
    %106 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %107 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn2.running_mean.npy"} : () -> tensor<256xf32>
    %108 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn2.running_var.npy"} : () -> tensor<256xf32>
    %109 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer3___1___bn2.num_batches_tracked.npy"} : () -> tensor<i64>
    %110 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn1.running_mean.npy"} : () -> tensor<512xf32>
    %111 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn1.running_var.npy"} : () -> tensor<512xf32>
    %112 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %113 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn2.running_mean.npy"} : () -> tensor<512xf32>
    %114 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn2.running_var.npy"} : () -> tensor<512xf32>
    %115 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___bn2.num_batches_tracked.npy"} : () -> tensor<i64>
    %116 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___downsample_1.running_mean.npy"} : () -> tensor<512xf32>
    %117 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___downsample_1.running_var.npy"} : () -> tensor<512xf32>
    %118 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___0___downsample_1.num_batches_tracked.npy"} : () -> tensor<i64>
    %119 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn1.running_mean.npy"} : () -> tensor<512xf32>
    %120 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn1.running_var.npy"} : () -> tensor<512xf32>
    %121 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn1.num_batches_tracked.npy"} : () -> tensor<i64>
    %122 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn2.running_mean.npy"} : () -> tensor<512xf32>
    %123 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn2.running_var.npy"} : () -> tensor<512xf32>
    %124 = "llh.weight"() {"weight_file" = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-21T06:52:50.599451+08:00/getattr_L__self___layer4___1___bn2.num_batches_tracked.npy"} : () -> tensor<i64>
    %125 = "llh.symbolic_int"() {"value" = "s0"} : () -> i64
    %126 = "llh.symbolic_int"() {"value" = "s2"} : () -> i64
    "llh.symbolic_bind"(%2, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 3, s1, s1)>} : (tensor<?x3x?x?xf32>, i64, i64) -> ()
    %127 = "llh.conv"(%2, %3) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 3, 3, 3, 3>, "group" = 1 : i64, "kernel_shape" = array<i64: 7, 7>, "stride" = array<i64: 2, 2>} : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%127, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 2) + 1), (((s1 + -1) floordiv 2) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %128 = "llh.batch_norm"(%127, %4, %5, %65, %66) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%128, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 2) + 1), (((s1 + -1) floordiv 2) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %129 = "llh.relu"(%128) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%129, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 2) + 1), (((s1 + -1) floordiv 2) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %130 = "llh.max_pool"(%129) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 2, 2>, "ceil_mode" = false} : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%130, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %131 = "llh.conv"(%130, %6) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%131, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %132 = "llh.batch_norm"(%131, %7, %8, %68, %69) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%132, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %133 = "llh.relu"(%132) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%133, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %134 = "llh.conv"(%133, %9) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%134, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %135 = "llh.batch_norm"(%134, %10, %11, %71, %72) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%135, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %136 = "llh.add"(%135, %130) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%136, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %137 = "llh.relu"(%136) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%137, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %138 = "llh.conv"(%137, %12) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%138, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %139 = "llh.batch_norm"(%138, %13, %14, %74, %75) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%139, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %140 = "llh.relu"(%139) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%140, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %141 = "llh.conv"(%140, %15) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x64x?x?xf32>, tensor<64x64x3x3xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%141, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %142 = "llh.batch_norm"(%141, %16, %17, %77, %78) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x64x?x?xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%142, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %143 = "llh.add"(%142, %137) : (tensor<?x64x?x?xf32>, tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%143, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %144 = "llh.relu"(%143) : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
    "llh.symbolic_bind"(%144, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 64, (((s1 + -1) floordiv 4) + 1), (((s1 + -1) floordiv 4) + 1))>} : (tensor<?x64x?x?xf32>, i64, i64) -> ()
    %145 = "llh.conv"(%144, %18) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 2, 2>} : (tensor<?x64x?x?xf32>, tensor<128x64x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%145, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %146 = "llh.batch_norm"(%145, %19, %20, %80, %81) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%146, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %147 = "llh.relu"(%146) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%147, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %148 = "llh.conv"(%147, %21) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%148, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %149 = "llh.batch_norm"(%148, %22, %23, %83, %84) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%149, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %150 = "llh.conv"(%144, %24) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 0, 0, 0, 0>, "group" = 1 : i64, "kernel_shape" = array<i64: 1, 1>, "stride" = array<i64: 2, 2>} : (tensor<?x64x?x?xf32>, tensor<128x64x1x1xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%150, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %151 = "llh.batch_norm"(%150, %25, %26, %86, %87) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%151, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %152 = "llh.add"(%149, %151) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%152, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %153 = "llh.relu"(%152) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%153, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %154 = "llh.conv"(%153, %27) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%154, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %155 = "llh.batch_norm"(%154, %28, %29, %89, %90) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%155, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %156 = "llh.relu"(%155) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%156, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %157 = "llh.conv"(%156, %30) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x128x?x?xf32>, tensor<128x128x3x3xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%157, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %158 = "llh.batch_norm"(%157, %31, %32, %92, %93) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%158, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %159 = "llh.add"(%158, %153) : (tensor<?x128x?x?xf32>, tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%159, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %160 = "llh.relu"(%159) : (tensor<?x128x?x?xf32>) -> tensor<?x128x?x?xf32>
    "llh.symbolic_bind"(%160, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 128, (((s1 + -1) floordiv 8) + 1), (((s1 + -1) floordiv 8) + 1))>} : (tensor<?x128x?x?xf32>, i64, i64) -> ()
    %161 = "llh.conv"(%160, %33) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 2, 2>} : (tensor<?x128x?x?xf32>, tensor<256x128x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%161, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %162 = "llh.batch_norm"(%161, %34, %35, %95, %96) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%162, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %163 = "llh.relu"(%162) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%163, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %164 = "llh.conv"(%163, %36) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%164, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %165 = "llh.batch_norm"(%164, %37, %38, %98, %99) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%165, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %166 = "llh.conv"(%160, %39) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 0, 0, 0, 0>, "group" = 1 : i64, "kernel_shape" = array<i64: 1, 1>, "stride" = array<i64: 2, 2>} : (tensor<?x128x?x?xf32>, tensor<256x128x1x1xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%166, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %167 = "llh.batch_norm"(%166, %40, %41, %101, %102) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%167, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %168 = "llh.add"(%165, %167) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%168, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %169 = "llh.relu"(%168) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%169, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %170 = "llh.conv"(%169, %42) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%170, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %171 = "llh.batch_norm"(%170, %43, %44, %104, %105) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%171, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %172 = "llh.relu"(%171) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%172, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %173 = "llh.conv"(%172, %45) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x256x?x?xf32>, tensor<256x256x3x3xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%173, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %174 = "llh.batch_norm"(%173, %46, %47, %107, %108) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x256x?x?xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%174, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %175 = "llh.add"(%174, %169) : (tensor<?x256x?x?xf32>, tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%175, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %176 = "llh.relu"(%175) : (tensor<?x256x?x?xf32>) -> tensor<?x256x?x?xf32>
    "llh.symbolic_bind"(%176, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 256, (((s1 + -1) floordiv 16) + 1), (((s1 + -1) floordiv 16) + 1))>} : (tensor<?x256x?x?xf32>, i64, i64) -> ()
    %177 = "llh.conv"(%176, %48) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 2, 2>} : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%177, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %178 = "llh.batch_norm"(%177, %49, %50, %110, %111) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%178, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %179 = "llh.relu"(%178) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%179, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %180 = "llh.conv"(%179, %51) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%180, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %181 = "llh.batch_norm"(%180, %52, %53, %113, %114) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%181, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %182 = "llh.conv"(%176, %54) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 0, 0, 0, 0>, "group" = 1 : i64, "kernel_shape" = array<i64: 1, 1>, "stride" = array<i64: 2, 2>} : (tensor<?x256x?x?xf32>, tensor<512x256x1x1xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%182, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %183 = "llh.batch_norm"(%182, %55, %56, %116, %117) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%183, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %184 = "llh.add"(%181, %183) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%184, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %185 = "llh.relu"(%184) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%185, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %186 = "llh.conv"(%185, %57) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%186, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %187 = "llh.batch_norm"(%186, %58, %59, %119, %120) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%187, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %188 = "llh.relu"(%187) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%188, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %189 = "llh.conv"(%188, %60) {"dilation" = array<i64: 1, 1>, "pad" = array<i64: 1, 1, 1, 1>, "group" = 1 : i64, "kernel_shape" = array<i64: 3, 3>, "stride" = array<i64: 1, 1>} : (tensor<?x512x?x?xf32>, tensor<512x512x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%189, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %190 = "llh.batch_norm"(%189, %61, %62, %122, %123) {"epsilon" = 1.000000e-05 : f64, "momentum" = 1.000000e-01 : f64} : (tensor<?x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%190, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %191 = "llh.add"(%190, %185) : (tensor<?x512x?x?xf32>, tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%191, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %192 = "llh.relu"(%191) : (tensor<?x512x?x?xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%192, %125, %126) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, (((s1 + -1) floordiv 32) + 1), (((s1 + -1) floordiv 32) + 1))>} : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    %193 = "llh.adaptive_average_pool"(%192) : (tensor<?x512x?x?xf32>) -> tensor<?x512x1x1xf32>
    "llh.symbolic_bind"(%193, %125) {"expressions" = affine_map<()[s0, s1] -> (s0, 512, 1, 1)>} : (tensor<?x512x1x1xf32>, i64) -> ()
    %194 = "llh.constant"() {"value" = 1 : i64} : () -> i64
    %195 = "llh.flatten"(%193, %194) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>
    "llh.symbolic_bind"(%195, %125) {"expressions" = affine_map<()[s0, s1] -> (s0, 512)>} : (tensor<?x512xf32>, i64) -> ()
    %196 = "llh.transpose"(%63) {"perms" = array<i64: 1, 0>} : (tensor<1000x512xf32>) -> tensor<512x1000xf32>
    %197 = "llh.matmul"(%195, %196) : (tensor<?x512xf32>, tensor<512x1000xf32>) -> tensor<?x1000xf32>
    "llh.symbolic_bind"(%197, %125) {"expressions" = affine_map<()[s0, s1] -> (s0, 1000)>} : (tensor<?x1000xf32>, i64) -> ()
    func.return %197 : tensor<?x1000xf32>
  }
}
