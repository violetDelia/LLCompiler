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
  * window stride类代码生成
* op完善

  * slice
* 添加AOT算子

  * 重写编译逻辑
* 支持GPU后端
* reduce fusion
* 量化指令调研和实现   L3 i4 --> L1/L2 f16
* kv chach 有必要在编译器写吗？

## log

2024.11.27:   代数化简: reshape(const) --> const
