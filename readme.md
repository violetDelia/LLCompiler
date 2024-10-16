# install

node:默认生成器是ninja ，其他生成器自行设置

安装
    ' pip install .'

开发模式：

 ' pip install -e .'

测试:
ninja check-llcompiler
ninja check-llc-python

# Todo

* 运行resnet [ 重写binary op infer 以及 reshape/empty ] 【添加 relation 表达 非必要】
* llh 基于symbol  的   to linalg 扩展
* affine -> vector -> x86/riscv
* gcu 以及luanch/ 线程选择？
* 华为的自动融合策略实现
* 量化指令调研和实现   L3 i4 --> L1/L2 f16
* 权重重排，推理优化 【类似dma合并？】
* flashattension + kv chach 编译器表达？
