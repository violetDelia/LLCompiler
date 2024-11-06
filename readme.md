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

* 动态性支持

  * conv 的动态代码生成
  * pool 动态代码生成
  * 支持动态的linaga 融合
* op完善

  * 训练的bn
* affine -> vector -> x86/riscv
* 添加AOT算子

  * 重写编译逻辑
* 支持GPU后端
* transform 自定义Pipeline
* 华为的自动融合策略实现
* 量化指令调研和实现   L3 i4 --> L1/L2 f16
* flashattension + kv chach
