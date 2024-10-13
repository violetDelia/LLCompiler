# install
node:默认生成器是ninja ，其他生成器自行设置

安装
    ' pip install .'

开发模式：

 ' pip install -e .'

测试:
ninja check-llcompiler
ninja check-llc-python
# note

编译器不进行内存的分配，因此运行的内存由Engine获取，编译器只考虑L2，L1内存的数据流。

Engine 负责执行编译好的模型文件。而且一个engine只负责运行一个函数。

python端的ExecutionEngine只具有内存预分配和执行Engine的功能。

python端的importer 负责构建xdsl 的mlir

do_compiler 负责将xdsl mlir 编译好，返回Engine
