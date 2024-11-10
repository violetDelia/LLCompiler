# 用transform dialect include构建pass

## 引言

transform dialect 在mlir在mlir中是十分特殊的dialect，它提供了另一种相比于手写Pass更加细粒度的IR转换的方式。并且可以将转换规则编写成库，像头文件一样引入。这样可以避免经常的修改Pass和Pipeline。虽然用transform dialect 写出的所有变换都可以用写成相应的Pass，但是采用transform dialect写的话可以避免重新编译，而且在一些比较复杂的优化场景上更加灵活。

而且在mlir的transform dialect实现中具备很多pass【标准】所不具备的一些功能。

## 加载transform include

mlir 中 加载了
